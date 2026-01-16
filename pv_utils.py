import re
import json
import ast

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

# Code_set = ["PartnershipProvider", "InfoGive", "InfoSeek", "PartnershipPatient", "InfoGiveSDOH", "SharedDecisionPatient", "Socioemotional/Empathy", "SharedDecisionProvider", "InfoSeekSDOH"]

# Sub_Code_set = ["salutation", "SchedulingAppt", "connection", "statePreferences", "Appreciation/Gratitude", "HealthCareAccessAndQuality", "signoff", "Instruction", "Generalinformation", "ApprovalofDecision", "NeighborhoodAndBuiltEnvironment", "Diagnostics", "carecoordination", "Drugs", "Symptoms", "maintainCommunication", "AcknowledgeSituation", "activeParticipation/involvement", "inviteCollaboration", "expressOpinions", "ExpressConcern/unease", "EncourageQuestions", "SummarizeConfirmUnderstanding", "Reassurance", "Approval", "Alignment", "SeekingApproval", "checkingUnderstanding", "EconomicStability", "ExploreOptions", "ShareOptions", "Sadness/fear", "Prognosis", "build trust", "checkingUnderstanding/clarification", "ApprovalofDecision/Reinforcement", "Approval/Reinforcement", "MakeDecision", "PositiveRemarks"]

Code_set = [
    "CareCoordinationPatient",
    "CareCoordinationProvider",
    "PartnershipPatient",
    "PartnershipProvider",
    "SDOH",
    "SharedDecisionPatient",
    "SharedDecisionProvider",
    "SocioEmotionalBehaviour",
]

Sub_Code_set = [
    "Appreciation/Gratitude",
    "Approval/Reinforcement",
    "ApprovalofDecision/Reinforcement",
    "Clinical Care",
    "EconomicStability",
    "EducationAccessAndQuality",
    "ExploreOptions",
    "HealthCareAccessAndQuality",
    "MakeDecision",
    "NeighborhoodAndBuiltEnvironment",
    "SeekingApproval",
    "ShareOptions",
    "SocialAndCommunityContext",
    "acknowledgePatientExpertiseKnowledge",
    "activeParticipation/involvement",
    "alignment",
    "build trust",
    "checkingUnderstanding/clarification",
    "connection",
    "expressOpinions",
    "inviteCollabration",
    "maintainCommunication",
    "requestsForOpinion",
    "salutation",
    "signoff",
    "statePreferences",
]


def is_valid_format(input_str):
    pattern = r"""
        ^\{\s*                         # Starts with {, allows spaces before and after
        "results"\s*:\s*\[              # The "results" key, followed by [
        (\s*\{                          # Starts the JSON object
            \s*"Code"\s*:\s*'[^']+'\s*, # "Code" field, must be a non-empty string
            \s*"Sub-code"\s*:\s*'[^']+'\s*, # "Sub-code" field, must be a non-empty string
            \s*"Span"\s*:\s*'[^']+'\s*  # "Span" field, must be a non-empty string
        \}\s*,?)*                       # Multiple objects are allowed, commas are optional
        \s*\]                           # End list
        \s*\}$                          # End dict
    """

    return bool(re.match(pattern, input_str, re.VERBOSE | re.DOTALL))

def fix_structure(input_str):
    # input_str = replace_str(input_str)

    
    # Use regular expression to extract complete dictionary object
    pattern = re.compile(r'\{\s*"Code"\s*:\s*"[^"]+"\s*,\s*"Sub-code"\s*:\s*"[^"]+"\s*,\s*"Span"\s*:\s*"[^"]+"\s*\}')
    matches = pattern.findall(input_str)
    
    # Reconstruct the results list to conform to the format
    corrected_results = ', '.join(matches)
    corrected_json = '{"results": [' + corrected_results + ']}'
    
    return corrected_json
    

# def safe_json_loads(input_str):

#     try:
#         return json.loads(input_str)
#     except json.JSONDecodeError:
#         # 尝试修复非法反斜杠
#         input_str = input_str.replace('\\', '\\\\')  # 全部转义
#         try:
#             return json.loads(input_str)
#         except json.JSONDecodeError:
#             # 尝试只保留符合结构的结果字段
#             pattern = re.compile(r'\{\s*"Code"\s*:\s*"[^"]*"\s*,\s*"Sub-code"\s*:\s*"[^"]*"\s*,\s*"Span"\s*:\s*"[^"]*"\s*\}')
#             matches = pattern.findall(input_str)
#             if not matches:
#                 return {"results": []}
#             return {"results": [json.loads(m.replace('\\', '\\\\')) for m in matches]}

def safe_json_loads(input_str):
    import html

    if isinstance(input_str, dict):
        return input_str  # already parsed

    try:
        return json.loads(input_str)
    except json.JSONDecodeError:
        pass

    # 尝试解码 HTML 实体符号并转义单引号
    input_str = html.unescape(input_str)
    input_str = input_str.replace("'", '"')  # 单引号改成双引号
    input_str = input_str.replace("\\", "\\\\")  # 再次防止反斜杠问题

    # 提取有效的 JSON 对象片段
    pattern = re.compile(r'\{\s*"Code"\s*:\s*"[^"]*"\s*,\s*"Sub-code"\s*:\s*"[^"]*"\s*,\s*"Span"\s*:\s*"[^"]*"\s*\}')
    matches = pattern.findall(input_str)

    results = []
    for m in matches:
        try:
            obj = json.loads(m)
            results.append(obj)
        except json.JSONDecodeError:
            continue

    return {"results": results}



def evaluate_eppc(items):
    """ pass the parameters"""
   
    return items


def evaluate_eppc_agg(items):
    true_answer = [item[0] for item in items]
    pred_answer = [item[1] for item in items]
    

    ## code
    true_codes = []
    pred_codes = []

    ## sub-code
    true_sub_codes = []
    pred_sub_codes = []

    ## span
    true_spans = []
    pred_spans = []
    # for t_answer, p_answer in zip(true_answer, pred_answer):
    #     # t_answer = ast.literal_eval(t_answer)
    #     t_answer = safe_json_loads(t_answer)
        
    #     if not is_valid_format(p_answer):
    #         p_answer = fix_structure(p_answer)
    #     p_answer = safe_json_loads(p_answer)
    
    for t_answer, p_answer in zip(true_answer, pred_answer):
        t_answer = safe_json_loads(t_answer)
        p_answer = safe_json_loads(p_answer)

        
        code = [anno.get("Code") for anno in t_answer.get("results")]
        pred_code = [pred.get("Code") for pred in p_answer.get("results")]
        true_codes.append(code)
        pred_codes.append(pred_code)

        sub_code = [anno.get("Sub-code") for anno in t_answer.get("results")]
        pred_sub_code = [pred.get("Sub-code") for pred in p_answer.get("results")]
        true_sub_codes.append(sub_code)
        pred_sub_codes.append(pred_sub_code)

        span = [anno.get("Span") for anno in t_answer.get("results")]
        extracted_span = [pred.get("Span") for pred in p_answer.get("results")]
        true_spans.append(span)
        pred_spans.append(extracted_span)

    
    precision_code, recall_code, f1_code = calculate_code(true_codes, pred_codes)
    precision_subcode, recall_subcode, f1_subcode = calculate_subcode(true_sub_codes, pred_sub_codes)
    precision_span, recall_span, f1_span = relaxed_match_evaluation_with_full_containment(true_spans, pred_spans, jaccard_threshold=0.6)
    
    return {"code": {"P": round(precision_code,4), "R": round(recall_code,4), "f1": round(f1_code,4)},
            "sub-code": {"P": round(precision_subcode,4), "R": round(recall_subcode,4), "f1": round(f1_subcode,4)},
            "span": {"P": round(precision_span,4), "R": round(recall_span,4), "f1": round(f1_span,4)}}



def calculate_code(true_codes, pred_codes):
    code_mlb = MultiLabelBinarizer(classes=Code_set)
    true_code_binary = code_mlb.fit_transform(true_codes)
    pred_code_binary = code_mlb.transform(pred_codes) 

    precision_code = precision_score(true_code_binary, pred_code_binary, average='micro')
    recall_code = recall_score(true_code_binary, pred_code_binary, average='micro')
    f1_code = f1_score(true_code_binary, pred_code_binary, average='micro')

    return precision_code, recall_code, f1_code

def calculate_subcode(true_sub_codes, pred_sub_codes):
    sub_code_mlb = MultiLabelBinarizer(classes=Sub_Code_set)
    true_subcode_binary = sub_code_mlb.fit_transform(true_sub_codes)
    pred_subcode_binary = sub_code_mlb.transform(pred_sub_codes)

    precision_subcode = precision_score(true_subcode_binary, pred_subcode_binary, average='micro')
    recall_subcode = recall_score(true_subcode_binary, pred_subcode_binary, average='micro')
    f1_subcode = f1_score(true_subcode_binary, pred_subcode_binary, average='micro')

    return precision_subcode, recall_subcode, f1_subcode

def calculate_jaccard_for_tokens(phrase1, phrase2):
    """
    Calculate the Jaccard coefficient of two phrases (based on tokens)
    :param phrase1: first phrase string
    :param phrase2: second phrase string
    :return: Jaccard coefficient (0~1)
    """
    set1 = set(phrase1.lower().split())
    set2 = set(phrase2.lower().split())

    # Computing intersection and union
    intersection = set1.intersection(set2)
    union = set1.union(set2)

    # Calculate the Jaccard coefficient
    if len(union) == 0:
        return 0
    return len(intersection) / len(union)


def is_full_containment_match(phrase1, phrase2):
    """
    Determine if phrase1 is completely contained in phrase2
    :param phrase1: first phrase (str) - true phrase
    :param phrase2: second phrase (str) - predicted phrase
    :return: True if phrase1 is completely contained in phrase2
    """
    set1 = set(phrase1.lower().split())
    set2 = set(phrase2.lower().split())
    
    # Determine whether set1 is a subset of set2
    return set1.issubset(set2)


def relaxed_match_evaluation_with_full_containment(true_entities_list, pred_entities_list, jaccard_threshold=0.6):
    """
    Evaluate partial matches of named entities using Relaxed Match, combining full containment logic and Jaccard coefficient
    :param true_entities_list: list of true entities, one per sentence [[str, ...], ...]
    :param pred_entities_list: list of predicted entities, one per sentence [[str, ...], ...]
    :param jaccard_threshold: Jaccard coefficient threshold for partial matches
    :return: Precision, Recall, F1 Score
    """
    true_positive = 0
    false_positive = 0
    false_negative = 0

    # Iterate over the true and predicted entities in each sentence
    for true_entities, pred_entities in zip(true_entities_list, pred_entities_list):
        matched_true = [False] * len(true_entities)
        matched_pred = [False] * len(pred_entities)

        # For each predicted entity, check whether it partially matches a real entity
        for i, pred_entity in enumerate(pred_entities):
            for j, true_entity in enumerate(true_entities):
                if not matched_true[j] and not matched_pred[i]:
                    # If the real entity is completely contained in the predicted entity, or the predicted entity is completely contained in the real entity, it is considered a match.
                    if is_full_containment_match(true_entity, pred_entity) or is_full_containment_match(pred_entity, true_entity):
                        true_positive += 1
                        matched_true[j] = True
                        matched_pred[i] = True
                    # Otherwise, use the Jaccard coefficient for partial matching evaluation.
                    elif calculate_jaccard_for_tokens(pred_entity, true_entity) >= jaccard_threshold:
                        true_positive += 1
                        matched_true[j] = True
                        matched_pred[i] = True

        # Counting False Positives and False Negatives
        false_positive += matched_pred.count(False)  # No matching predicted entities found
        false_negative += matched_true.count(False)  # No real entity matched

    # Calculate Precision, Recall, F1 Score
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1



