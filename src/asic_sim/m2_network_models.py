from __future__ import annotations

import argparse

from .formatting import fmt_rate, fmt_time_s
from .hardware import HardwareSpec, get_hardware
from .m2 import DEFAULT_CANDIDATES, _M2Simulator, simulate_m2
from .models import ModelSpec, get_model


DEFAULT_CONCURRENCY = (1, 8, 64)


class _CutThroughSimulator(_M2Simulator):
    """Optimistic cut-through counterpart to the conservative M2 network.

    A packet may start on the next directed link after its header reaches the
    next router instead of waiting for the complete payload to arrive. Each
    link is still a finite FIFO serialization resource. Backpressure between
    adjacent links and virtual-channel effects are intentionally omitted, so
    this model is an optimistic bracket rather than a claim about a concrete
    NoC implementation.
    """

    def _handle_packet_hop(
        self,
        time_s: float,
        packet_id: int,
        packet_start_s: float,
        path: tuple[int, ...],
        hop_index: int,
        payload_bytes: float,
        callback: str,
        cb_data: tuple[object, ...],
    ) -> None:
        del packet_id
        if hop_index >= len(path) - 1:
            self.packet_latencies.append(time_s - packet_start_s)
            self.push(time_s, callback, *cb_data)
            return

        link = (path[hop_index], path[hop_index + 1])
        start = max(time_s, self.link_available.get(link, 0.0))
        wait = start - time_s
        serialization = payload_bytes / self.link_bandwidth_bytes_s
        link_finish = start + serialization

        self.link_available[link] = link_finish
        self.link_busy[link] = self.link_busy.get(link, 0.0) + serialization
        self.link_wait_total += wait
        self.link_wait_samples.append(wait)

        last_hop = hop_index == len(path) - 2
        if last_hop:
            # The callback consumes the full payload, so completion occurs when
            # the tail leaves the final link plus one router traversal.
            next_time = link_finish + self.router_latency_s
        else:
            # Cut-through: the header can request the next link before the
            # current link has finished serializing the packet tail.
            next_time = start + self.router_latency_s

        self.push(
            next_time,
            "packet_hop",
            -1,
            packet_start_s,
            path,
            hop_index + 1,
            payload_bytes,
            callback,
            cb_data,
        )


def simulate_m2_cut_through(
    model: ModelSpec,
    hardware: HardwareSpec,
    *,
    cluster_size: int,
    link_bandwidth_gb_s: float,
    concurrency: int,
    bits_per_weight: float = 4.0,
    shared_expert_bits: float | None = None,
    overhead_fraction: float = 0.05,
    activation_bits: float = 16.0,
    profile: str = "balanced",
    seed: int = 42,
    issue_spacing_s: float = 0.0,
):
    shared_bits = bits_per_weight if shared_expert_bits is None else shared_expert_bits
    return _CutThroughSimulator(
        model,
        hardware,
        cluster_size=cluster_size,
        link_bandwidth_gb_s=link_bandwidth_gb_s,
        bits_per_weight=bits_per_weight,
        shared_expert_bits=shared_bits,
        overhead_fraction=overhead_fraction,
        activation_bits=activation_bits,
        concurrency=concurrency,
        profile=profile,
        seed=seed,
        issue_spacing_s=issue_spacing_s,
    ).run()


def _parse_candidate(value: str) -> tuple[int, float]:
    try:
        cluster_text, bandwidth_text = value.split(":", 1)
        cluster = int(cluster_text)
        bandwidth = float(bandwidth_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("candidate must look like CLUSTER:GB/s, e.g. 16:64") from exc
    if cluster <= 0 or bandwidth <= 0:
        raise argparse.ArgumentTypeError("cluster and bandwidth must be positive")
    return cluster, bandwidth


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asic-sim-m2-net",
        description="Bracket M2 NoC timing with store-and-forward and optimistic cut-through routing.",
    )
    parser.add_argument("--model", default="kimi-k3")
    parser.add_argument("--hardware", default="fabric-64x32")
    parser.add_argument("--bits", type=float, default=4.0)
    parser.add_argument("--shared-bits", type=float, default=16.0)
    parser.add_argument("--overhead", type=float, default=0.05)
    parser.add_argument("--activation-bits", type=float, default=16.0)
    parser.add_argument("--candidate", type=_parse_candidate, action="append", dest="candidates")
    parser.add_argument("--concurrency", type=int, nargs="+", default=list(DEFAULT_CONCURRENCY))
    parser.add_argument("--profile", choices=("balanced", "hot", "zipf"), default="balanced")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--issue-spacing-us", type=float, default=0.0)
    return parser


def _run(args: argparse.Namespace) -> int:
    model = get_model(args.model)
    hardware = get_hardware(args.hardware)
    candidates = tuple(args.candidates) if args.candidates else DEFAULT_CANDIDATES
    concurrency_levels = tuple(sorted(set(args.concurrency)))
    if not concurrency_levels or any(value <= 0 for value in concurrency_levels):
        raise ValueError("concurrency values must be positive")
    if args.issue_spacing_us < 0:
        raise ValueError("issue spacing cannot be negative")

    print(f"{model.name} -> {hardware.name}")
    print("  purpose:                bracket packet-network timing before adding compute/KV")
    print("  pessimistic model:      store-and-forward XY, finite FIFO links")
    print("  optimistic model:       cut-through XY, finite FIFO links, no backpressure")
    print(f"  base/shared precision:  {args.bits:g} / {args.shared_bits:g} bit")
    print(f"  routing profile:        {args.profile}")
    print(f"  issue spacing:          {args.issue_spacing_us:g} us")
    print("  candidates:             " + ", ".join(f"C{c}@{bw:g}GB/s" for c, bw in candidates))

    print("\nNETWORK-MODEL BRACKET")
    print(
        "CLUSTER LINK   CONC   SF TPS       CT TPS       SF RET  CT RET  "
        "SF P95 LAT  CT P95 LAT  CT LINK  CT MAXQ    CT P95 BARRIER"
    )

    for cluster_size, bandwidth in candidates:
        for concurrency in concurrency_levels:
            common = dict(
                cluster_size=cluster_size,
                concurrency=concurrency,
                bits_per_weight=args.bits,
                shared_expert_bits=args.shared_bits,
                overhead_fraction=args.overhead,
                activation_bits=args.activation_bits,
                profile=args.profile,
                seed=args.seed,
                issue_spacing_s=args.issue_spacing_us * 1e-6,
            )
            baseline = simulate_m2_cut_through(
                model,
                hardware,
                link_bandwidth_gb_s=1e9,
                **common,
            )
            sf = simulate_m2(
                model,
                hardware,
                link_bandwidth_gb_s=bandwidth,
                **common,
            )
            ct = simulate_m2_cut_through(
                model,
                hardware,
                link_bandwidth_gb_s=bandwidth,
                **common,
            )
            if not (sf.resident and ct.resident and baseline.resident):
                print(f"C{cluster_size:<6} {bandwidth:>4g}G {concurrency:>6}   NOFIT")
                continue
            sf_retain = sf.throughput_tps / baseline.throughput_tps if baseline.throughput_tps else 0.0
            ct_retain = ct.throughput_tps / baseline.throughput_tps if baseline.throughput_tps else 0.0
            print(
                f"C{cluster_size:<6} {bandwidth:>4g}G {concurrency:>6}  "
                f"{fmt_rate(sf.throughput_tps):>11}  {fmt_rate(ct.throughput_tps):>11}  "
                f"{sf_retain:>6.1%}  {ct_retain:>6.1%}  "
                f"{fmt_time_s(sf.p95_latency_s):>10}  {fmt_time_s(ct.p95_latency_s):>10}  "
                f"{ct.hottest_link_utilization:>7.1%}  {fmt_time_s(ct.max_link_queue_wait_s):>9}  "
                f"{fmt_time_s(ct.p95_barrier_tail_s):>14}"
            )

    print("\nHOW TO READ THIS")
    print("  SF and CT bracket the packet-forwarding assumption; neither is yet a cycle-accurate NoC.")
    print("  If CT RET is near 100% but SF RET is low, serialization-per-hop latency dominates the old result, not bandwidth saturation.")
    print("  If both RET values fall with rising concurrency and queue wait grows, the candidate link rate is genuinely underprovisioned.")
    print("  The realistic design should ultimately land between these bounds after flits, credits/backpressure and virtual channels are modeled.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
