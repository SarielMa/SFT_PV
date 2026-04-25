import json
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

# =========================
# CONFIG
# =========================

jsonl_path = "EPPC_3.1_70B.jsonl"

# Code_set = [
#     "InformationGiving",
#     "InformationSeeking",
#     "SDOHInformationGiving",
#     "SDOHInformationSeeking",
#     "PartnershipPatient",
#     "PartnershipProvider",
#     "SharedDecisionPatient",
#     "SharedDecisionProvider",
#     "SocioEmotionalBehaviour",
# ]

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

# Sub_Code_set = [
#     "salutation",
#     "schedulingAppointment",
#     "inviteCollaboration",
#     "connection",
#     "statePreferences",
#     "appreciationGratitude",
#     "healthCareAccessAndQuality",
#     "signoff",
#     "instruction",
#     "careCoordination",
#     "approvalOfDecisionReinforcement",
#     "neighborhoodAndBuiltEnvironment",
#     "generalInformation",
#     "diagnostics",
#     "drugs",
#     "symptoms",
#     "maintainCommunication",
#     "acknowledgeError",
#     "activeParticipationInvolvement",
#     "expressOpinions",
#     "expressConcernUnease",
#     "encourageQuestions",
#     "summarizeConfirmUnderstanding",
#     "reassurance",
#     "approvalReinforcement",
#     "alignment",
#     "seekingApproval",
#     "checkingUnderstandingClarification",
#     "economicStability",
#     "exploreOptions",
#     "shareOptions",
#     "sadnessFear",
#     "buildTrust",
#     "prognosis",
#     "makeDecision",
#     "positiveRemarks",
# ]

# =========================
# HELPER FUNCTION
# =========================

def parse_results(json_string):
    """
    Parse evaluate_eppc item and return list of dicts
    """
    try:
        data = json.loads(json_string)
        return data.get("results", [])
    except:
        return []


# =========================
# LOAD DATA
# =========================

with open(jsonl_path, "r", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f]

# =========================
# STORAGE
# =========================

# For micro F1 per class
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

    # ASSUMPTION:
    gt_results = parse_results(eval_list[0])
    pred_results = parse_results(eval_list[1])

    # Convert to sets of labels
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
# METRIC COMPUTATION
# =========================

def compute_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return precision, recall, f1


print("\n================ CODE LEVEL =================")
for code in Code_set:
    p, r, f1 = compute_f1(
        code_tp[code],
        code_fp[code],
        code_fn[code]
    )
    print(f"{code:35s}  P={p:.4f}  R={r:.4f}  F1={f1:.4f}")

print("\n================ SUB-CODE LEVEL =================")
for sub in Sub_Code_set:
    p, r, f1 = compute_f1(
        sub_tp[sub],
        sub_fp[sub],
        sub_fn[sub]
    )
    print(f"{sub:40s}  P={p:.4f}  R={r:.4f}  F1={f1:.4f}")