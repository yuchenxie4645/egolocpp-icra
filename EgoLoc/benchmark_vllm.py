#!/usr/bin/env python3
"""
Benchmark a vLLM-served Qwen3.5-9B via the OpenAI-compatible API.

- 50 queries dispatched concurrently (asyncio)
- 30 unique essay prompts, cycled to fill 50
- ~4000 max_tokens per response
- Streams responses, measures TTFT, per-query tok/s, aggregate tok/s
- Prints everything to stdout
- Authenticates with a Bearer API key (read from VLLM_API_KEY env var)
"""

import asyncio
import json
import os
import statistics
import time

import httpx

# ---------- Config ----------
API_BASE = "http://localhost:8000/v1"
MODEL = "Qwen/Qwen3.5-9B"
MAX_TOKENS = 81920
TEMPERATURE = 0.7
TOP_P = 0.95
NUM_QUERIES = 50
NUM_PROMPTS = 30
REQUEST_TIMEOUT_S = 600
NUM_RUNS = 50                 # how many full benchmark runs to perform
SLEEP_BETWEEN_RUNS_S = 30       # pause between runs (seconds)

# API key for the vLLM endpoint. Read from env so it never lives in this file.
# Set it before running, e.g.:  export VLLM_API_KEY="your-key-here"
API_KEY = os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
# ----------------------------

PROMPTS = [
    "Write a detailed essay about the history and evolution of the internet, from ARPANET to the modern day.",
    "Explain the core principles of existentialist philosophy and discuss its major proponents.",
    "Analyze the multifaceted impact of climate change on coastal cities worldwide.",
    "Discuss the evolution of jazz music from its roots in New Orleans to its global influence.",
    "Write a comprehensive essay on the role and representation of women in STEM fields.",
    "Explain quantum computing to a beginner, covering qubits, superposition, and entanglement.",
    "Analyze Shakespeare's use of tragedy across his major works, with examples.",
    "Discuss the future of space exploration, including Mars missions and interstellar travel.",
    "Write about the psychology of decision-making and the cognitive biases that shape it.",
    "Explain the economic impact of globalization on developing nations.",
    "Analyze the rise of streaming platforms and their effect on traditional media.",
    "Discuss the ethical considerations surrounding artificial intelligence development.",
    "Write a detailed essay on the history, rise, and fall of the Roman Empire.",
    "Explain the science of black holes, including formation, event horizons, and Hawking radiation.",
    "Analyze the cultural impact of social media on modern society.",
    "Discuss the future of renewable energy and its role in combating climate change.",
    "Write about the art of Renaissance painting and its major masters.",
    "Explain Einstein's theory of relativity in accessible terms.",
    "Analyze the complex causes and consequences of World War I.",
    "Discuss the philosophy and practice of mindfulness in modern life.",
    "Write a comprehensive essay on the history and evolution of democracy.",
    "Explain the biology and applications of CRISPR gene editing.",
    "Analyze the impact of remote work on corporate culture and productivity.",
    "Discuss the evolution of language and the emergence of new words.",
    "Write about the mathematics behind modern cryptography.",
    "Explain the neuroscience of memory formation and recall.",
    "Analyze the rise of cryptocurrencies and their economic implications.",
    "Discuss the future of education in the digital age.",
    "Write about the anthropology of rituals across different cultures.",
    "Explain the chemistry of photosynthesis and its importance to life on Earth.",
]


async def send_query(client: httpx.AsyncClient, prompt: str, query_id: int) -> dict:
    start = time.perf_counter()
    first_token_time = None
    token_count = 0
    prompt_tokens = 0
    error = None

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    headers = {}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    try:
        async with client.stream(
            "POST",
            f"{API_BASE}/chat/completions",
            json=payload,
            headers=headers,
            timeout=REQUEST_TIMEOUT_S,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                # Mark time-to-first-token on the first content delta
                if first_token_time is None:
                    choices = chunk.get("choices") or []
                    if choices and (choices[0].get("delta") or {}).get("content"):
                        first_token_time = time.perf_counter()

                # Final chunk carries usage when include_usage=True
                usage = chunk.get("usage")
                if usage:
                    token_count = usage.get("completion_tokens", 0) or 0
                    prompt_tokens = usage.get("prompt_tokens", 0) or 0
    except Exception as e:  # surface the failure but don't kill the gather
        error = repr(e)

    end = time.perf_counter()
    gen_time = (end - first_token_time) if first_token_time else (end - start)
    total_time = end - start

    return {
        "query_id": query_id,
        "prompt_idx": query_id % NUM_PROMPTS,
        "tokens": token_count,
        "prompt_tokens": prompt_tokens,
        "ttft": (first_token_time - start) if first_token_time else None,
        "gen_time": gen_time,
        "total_time": total_time,
        "tokens_per_sec": (token_count / gen_time) if gen_time > 0 else 0.0,
        "error": error,
    }


async def run_once(client: httpx.AsyncClient, run_idx: int) -> dict:
    """Run a single benchmark pass; returns a per-run summary dict."""
    tasks = [
        send_query(client, PROMPTS[i % NUM_PROMPTS], i)
        for i in range(NUM_QUERIES)
    ]

    wall_start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    wall_end = time.perf_counter()

    ok = [r for r in results if not r["error"]]
    failed = [r for r in results if r["error"]]
    total_tokens = sum(r["tokens"] for r in ok)
    total_prompt_tokens = sum(r["prompt_tokens"] for r in ok)
    tps_list = [r["tokens_per_sec"] for r in ok if r["tokens_per_sec"] > 0]
    ttft_list = [r["ttft"] for r in ok if r["ttft"] is not None]

    summary = {
        "run_idx": run_idx,
        "wall_time": wall_end - wall_start,
        "successful": len(ok),
        "failed": len(failed),
        "total_tokens": total_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "aggregate_tps": total_tokens / (wall_end - wall_start) if (wall_end - wall_start) > 0 else 0.0,
        "mean_tps": statistics.mean(tps_list) if tps_list else 0.0,
        "median_tps": statistics.median(tps_list) if tps_list else 0.0,
        "mean_ttft": statistics.mean(ttft_list) if ttft_list else 0.0,
        "median_ttft": statistics.median(ttft_list) if ttft_list else 0.0,
        "results": results,
    }
    return summary


def print_run_summary(s: dict) -> None:
    print(f"\n--- Run {s['run_idx'] + 1}/{NUM_RUNS} ---")
    print(f"  Wall time:           {s['wall_time']:.2f}s")
    print(f"  Successful:          {s['successful']}/{NUM_QUERIES}")
    if s["failed"]:
        print(f"  Failed:              {s['failed']}")
    print(f"  Total output tokens: {s['total_tokens']}")
    print(f"  Aggregate tok/s:     {s['aggregate_tps']:.2f}")
    print(f"  Mean per-query tok/s:{s['mean_tps']:.2f}")
    print(f"  Median per-query tok/s: {s['median_tps']:.2f}")
    print(f"  Mean TTFT:           {s['mean_ttft']:.3f}s")


async def main() -> None:
    if not API_KEY:
        raise SystemExit(
            "VLLM_API_KEY (or OPENAI_API_KEY) is not set. "
            "Export it before running, e.g.:  export VLLM_API_KEY='your-key'"
        )

    print(f"Loop benchmark: {NUM_RUNS} runs | {NUM_QUERIES} queries/run | "
          f"{NUM_PROMPTS} prompts | max_tokens={MAX_TOKENS} | "
          f"sleep_between={SLEEP_BETWEEN_RUNS_S}s")
    print(f"Model: {MODEL} @ {API_BASE}")
    print(f"Auth: Bearer key (length={len(API_KEY)})")
    print("=" * 90)

    limits = httpx.Limits(max_connections=NUM_QUERIES, max_keepalive_connections=NUM_QUERIES)
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S, limits=limits) as client:
        all_runs: list[dict] = []
        loop_start = time.perf_counter()

        for run_idx in range(NUM_RUNS):
            run_summary = await run_once(client, run_idx)
            all_runs.append(run_summary)
            print_run_summary(run_summary)

            if run_idx < NUM_RUNS - 1 and SLEEP_BETWEEN_RUNS_S > 0:
                await asyncio.sleep(SLEEP_BETWEEN_RUNS_S)

        loop_end = time.perf_counter()

    # ----- Cross-run aggregate -----
    total_tokens_all = sum(r["total_tokens"] for r in all_runs)
    total_prompt_tokens_all = sum(r["total_prompt_tokens"] for r in all_runs)
    total_queries_all = sum(r["successful"] for r in all_runs)
    total_failed_all = sum(r["failed"] for r in all_runs)
    loop_wall = loop_end - loop_start

    aggregate_tps_per_run = [r["aggregate_tps"] for r in all_runs if r["aggregate_tps"] > 0]
    mean_tps_per_run = [r["mean_tps"] for r in all_runs if r["mean_tps"] > 0]
    mean_ttft_per_run = [r["mean_ttft"] for r in all_runs if r["mean_ttft"] > 0]

    print("\n" + "#" * 90)
    print(f"CROSS-RUN SUMMARY ({NUM_RUNS} runs)")
    print(f"  Total loop wall time:        {loop_wall:.2f}s")
    print(f"  Total successful queries:    {total_queries_all} / {NUM_RUNS * NUM_QUERIES}")
    if total_failed_all:
        print(f"  Total failed queries:        {total_failed_all}")
    print(f"  Total output tokens (all):   {total_tokens_all}")
    print(f"  Total prompt tokens (all):   {total_prompt_tokens_all}")
    print(f"  Overall aggregate tok/s:     {total_tokens_all / loop_wall:.2f}  "
          f"(total_tokens / loop_wall)")
    if aggregate_tps_per_run:
        print(f"  Mean run-aggregate tok/s:    {statistics.mean(aggregate_tps_per_run):.2f}")
        print(f"  Median run-aggregate tok/s:  {statistics.median(aggregate_tps_per_run):.2f}")
        if len(aggregate_tps_per_run) > 1:
            print(f"  Stdev run-aggregate tok/s:   {statistics.stdev(aggregate_tps_per_run):.2f}")
    if mean_tps_per_run:
        print(f"  Mean of per-run mean tok/s:  {statistics.mean(mean_tps_per_run):.2f}")
        print(f"  Mean of per-run median tok/s:{statistics.median(mean_tps_per_run):.2f}")
    if mean_ttft_per_run:
        print(f"  Mean of per-run mean TTFT:   {statistics.mean(mean_ttft_per_run):.3f}s")
    print("#" * 90)


if __name__ == "__main__":
    asyncio.run(main())
