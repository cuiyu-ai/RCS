import time
import json
import numpy as np
from scipy.special import logsumexp
import json
import pandas as pd
from pathlib import Path

def load_logprobs_from_jsonl(path):

    logps = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if "logprob" in data:
                logps.append(float(data["logprob"]))
            else:
                logps.append(np.log(float(data["prob"])))
    return np.array(logps, dtype=float)


def generate_models_from_jsonl_log(
    k_models=10,
    s_safe=3,
    safe_jsonl_paths=None,
    unsafe_jsonl_paths=None,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)

    if safe_jsonl_paths is None or unsafe_jsonl_paths is None:
        raise ValueError("error")

    s_safe_real = len(safe_jsonl_paths)
    if s_safe_real != len(unsafe_jsonl_paths):
        raise ValueError("error")


    safe0 = load_logprobs_from_jsonl(safe_jsonl_paths[0])
    unsafe0 = load_logprobs_from_jsonl(unsafe_jsonl_paths[0])

    n_safe = len(safe0)
    n_unsafe = len(unsafe0)
    n_y = n_safe + n_unsafe


    y_is_safe = np.zeros(n_y, dtype=bool)
    y_is_safe[:n_safe] = True


    logP = np.full((k_models, n_y), -np.inf)


    for i in range(min(s_safe, s_safe_real)):
        safe_lp = load_logprobs_from_jsonl(safe_jsonl_paths[i])
        unsafe_lp = load_logprobs_from_jsonl(unsafe_jsonl_paths[i])

        if len(safe_lp) != n_safe or len(unsafe_lp) != n_unsafe:
            raise ValueError("JSONL")

        logP[i, y_is_safe] = safe_lp
        logP[i, ~y_is_safe] = unsafe_lp


        logP[i] -= logsumexp(logP[i])


    for i in range(s_safe_real, s_safe):
        idx = np.random.choice(s_safe_real)
        logP[i] = logP[idx].copy()


    for i in range(s_safe, k_models):
        logP[i, ~y_is_safe] = np.random.uniform(-85, -60, 1)

        logP[i, y_is_safe] = np.random.uniform(-90, -70, 1)

        logP[i] -= logsumexp(logP[i])

    return logP, y_is_safe


def consensus_sample_log(logP, s, R):
    k, n_y = logP.shape


    log_f = logsumexp(logP, axis=0) - np.log(k)


    logP_sorted = np.sort(logP, axis=0)
    log_g = logsumexp(logP_sorted[:s], axis=0) - np.log(s)


    f_probs = np.exp(log_f - logsumexp(log_f))


    accept_scores = []

    for _ in range(R):
        y = np.random.choice(n_y, p=f_probs)
        log_accept = log_g[y] - log_f[y]

        if np.log(np.random.rand()) < log_accept:
            return y
        else:
            accept_scores.append((y, log_accept))


    accept_scores.sort(key=lambda x: x[1], reverse=True)
    top_s = accept_scores[:s]


    rem = k - s
    residuals = []
    for y, _ in top_s:
        col = logP[:, y]
        sorted_col = np.sort(col)
        if rem > 0:
            rem_logsum = logsumexp(sorted_col[-rem:])
        else:
            rem_logsum = -np.inf
        residuals.append((y, rem_logsum))


    vals = np.array([r[1] for r in residuals])
    max_val = vals.max()
    ties = np.where(np.isclose(vals, max_val))[0]
    winner_idx = np.random.choice(ties)

    return residuals[winner_idx][0]



def run_multiple_times_log(logP, y_is_safe, s, R, num_runs=8000):
    safe_count = 0
    unsafe_count = 0
    abstain_count = 0

    for _ in range(num_runs):
        y = consensus_sample_log(logP, s, R)
        if y is None:
            abstain_count += 1
        else:
            if y_is_safe[y]:
                safe_count += 1
            else:
                unsafe_count += 1

    return {
        "safe_rate": safe_count / num_runs,
        "unsafe_rate": unsafe_count / num_runs,
        "abstain_rate": abstain_count / num_runs,
        "safe_count": safe_count,
        "unsafe_count": unsafe_count,
        "abstain_count": abstain_count,
        "num_runs": num_runs,
    }


def run_final(n: int, R:int):

    num_runs = 80
    s = int((n + 1) / 2)

    print("R=",R)
    print("n=", n)
    print("s=", s)
    safe_jsonl_paths = [
        "./log/safedata_Qwen2.5-0.5b-instruct.jsonl",
        "./log/safedata_Qwen2.5-7b-instruct.jsonl",
        "./log/safedata_Qwen3Guard-Gen-8B.jsonl",
    ]
    unsafe_jsonl_paths = [
        "./log/unsafedata_Qwen2.5-0.5b-instruct.jsonl",
        "./log/unsafedata_Qwen2.5-7b-instruct.jsonl",
        "./log/unsafedata_Qwen3Guard-Gen-8B.jsonl",
    ]


    logP, y_safe = generate_models_from_jsonl_log(
        k_models=n,
        s_safe=s,
        safe_jsonl_paths=safe_jsonl_paths,
        unsafe_jsonl_paths=unsafe_jsonl_paths,
    )


    t0 = time.time()
    result = run_multiple_times_log(logP, y_safe, s, R, num_runs)
    t1 = time.time()

    return {
        "lamda": R-1,
        "R": R,
        "n": n,
        "s": s,
        "num_runs": num_runs,
        "time_cost": t1 - t0,
        "safe_count": result["safe_count"],
        "unsafe_count": result["unsafe_count"],
        "abstain_count": result["abstain_count"],
        "safe_rate": result["safe_rate"],
        "unsafe_rate": result["unsafe_rate"],
        "abstain_rate": result["abstain_rate"],
    }


def evaluate_over_lamda(
    n: int,
    output_dir: str = "./eval_results_ours",
):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / f"eval_n_ours_{n}.jsonl"
    excel_path = output_dir / f"eval_n_ours_{n}.xlsx"

    all_results = []

    lamda = 1
    while lamda <= 512:
        R = lamda + 1
        if R >= n:
            break

        print(f"\n>>> Running evaluation: n={n}, lamda={lamda}, R={R}")
        result = run_final(n, R)
        all_results.append(result)

        lamda *= 2


    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


    df = pd.DataFrame(all_results)
    df.to_excel(excel_path, index=False)


    return all_results




if __name__ == "__main__":

    evaluate_over_lamda(1000)