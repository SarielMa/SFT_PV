import json
from collections import defaultdict

# =========================
# CONFIG
# =========================

jsonl_path = "EPPC_3.1_70B.jsonl"

Code_set = [
    "InfoGive",
    "InfoGiveSDOH",
    "InfoSeek",
    "InfoSeekSDOH",
    "PartnershipPatient",
    "PartnershipProvider",
    "SharedDecisionPatient",
    "SharedDecisionProvider",
    "Socioemotional/Empathy",
]

Sub_Code_set = [
    "AcknowledgeSituation",
    "Alignment",
    "Appreciation/Gratitude",
    "Approval/Reinforcement",
    "ApprovalofDecision/Reinforcement",
    "Diagnostics",
    "Drugs",
    "EconomicStability",
    "EncourageQuestions",
    "ExploreOptions",
    "ExpressConcern/unease",
    "Generalinformation",
    "HealthCareAccessAndQuality",
    "Instruction",
    "MakeDecision",
    "NeighborhoodAndBuiltEnvironment",
    "PositiveRemarks",
    "Prognosis",
    "Reassurance",
    "Sadness/fear",
    "SchedulingAppt",
    "SeekingApproval",
    "ShareOptions",
    "SummarizeConfirmUnderstanding",
    "Symptoms",
    "activeParticipation/involvement",
    "build trust",
    "carecoordination",
    "checkingUnderstanding/clarification",
    "connection",
    "expressOpinions",
    "inviteCollaboration",
    "maintainCommunication",
    "salutation",
    "signoff",
    "statePreferences",
]

# =========================
# HELPER FUNCTION
# =========================

def parse_results(json_string):
    try:
        data = json.loads(json_string)
        return data.get("results", [])
    except:
        return []

def compute_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return precision, recall, f1


# =========================
# LOAD DATA
# =========================

with open(jsonl_path, "r", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f]

# =========================
# STORAGE
# =========================

code_tp = defaultdict(int)
code_fp = defaultdict(int)
code_fn = defaultdict(int)

sub_tp = defaultdict(int)
sub_fp = defaultdict(int)
sub_fn = defaultdict(int)

# =========================
# MAIN LOOP
# =========================

for sample in samples:

    eval_list = sample.get("evaluate_eppc", [])

    if len(eval_list) != 2:
        continue

    gt_results = parse_results(eval_list[0])
    pred_results = parse_results(eval_list[1])

    gt_codes = set()
    gt_subcodes = set()

    for item in gt_results:
        if "Code" in item:
            gt_codes.add(item["Code"])
        if "Sub-code" in item:
            gt_subcodes.add(item["Sub-code"])

    pred_codes = set()
    pred_subcodes = set()

    for item in pred_results:
        if "Code" in item:
            pred_codes.add(item["Code"])
        if "Sub-code" in item:
            pred_subcodes.add(item["Sub-code"])

    # ---- Code level ----
    for code in Code_set:
        if code in gt_codes and code in pred_codes:
            code_tp[code] += 1
        elif code not in gt_codes and code in pred_codes:
            code_fp[code] += 1
        elif code in gt_codes and code not in pred_codes:
            code_fn[code] += 1

    # ---- Sub-code level ----
    for sub in Sub_Code_set:
        if sub in gt_subcodes and sub in pred_subcodes:
            sub_tp[sub] += 1
        elif sub not in gt_subcodes and sub in pred_subcodes:
            sub_fp[sub] += 1
        elif sub in gt_subcodes and sub not in pred_subcodes:
            sub_fn[sub] += 1


# =========================
# PERFORMANCE
# =========================

print("\n================ SUB-CODE LEVEL =================")
for sub in Sub_Code_set:
    p, r, f1 = compute_f1(
        sub_tp[sub],
        sub_fp[sub],
        sub_fn[sub]
    )
    print(f"{sub:40s}  P={p:.4f}  R={r:.4f}  F1={f1:.4f}")


# =========================
# ✅ ERROR DISTRIBUTION (NEW)
# =========================

print("\n================ SUB-CODE ERROR DISTRIBUTION =================")

error_stats = []

for sub in Sub_Code_set:
    tp = sub_tp[sub]
    fp = sub_fp[sub]
    fn = sub_fn[sub]

    support = tp + fn
    total_error = fp + fn
    error_rate = total_error / support if support > 0 else 0

    error_stats.append((sub, support, tp, fp, fn, total_error, error_rate))

# sort by total error (descending)
error_stats.sort(key=lambda x: x[5], reverse=True)

for sub, support, tp, fp, fn, total_error, error_rate in error_stats:
    print(
        f"{sub:40s}  "
        f"Support={support:4d}  "
        f"TP={tp:4d}  FP={fp:4d}  FN={fn:4d}  "
        f"Error={total_error:4d}  "
        f"ErrRate={error_rate:.4f}"
    )