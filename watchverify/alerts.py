"""Which observations need a decision, and which prompt to raise.

Kept out of the interface so it can be tested and reasoned about on its own. Nothing here
decides anything about a person: it selects which question to put in front of a reviewer.
The application never contacts anyone — it asks, records the answer, and leaves the
original alert unchanged.
"""

# Retail-activity candidates prompt about police; fall and person-down candidates prompt
# about an ambulance. Categories outside this map (recovery, activity_clear) are state
# changes, not alerts, and must never raise a prompt.
ALERT_BRANCH = {
    "unusual_activity": "activity",
    "activity": "activity",
    "activity_candidate": "activity",
    "possible_fall": "fall",
    "person_down": "fall",
}

BRANCH_PROMPT = {
    "activity": {
        "heading": "Possible theft — your decision",
        "model": "activity",
        "call": ("police_called", "Call police"),
        "decline": ("police_not_called", "Do not call"),
        "fallback": ("This flags movement resembling clips labelled shoplifting. It does not "
                     "establish theft and does not identify anyone."),
    },
    "fall": {
        "heading": "Possible person down — your decision",
        "model": "fall",
        "call": ("ambulance_called", "Call ambulance"),
        "decline": ("ambulance_not_called", "Do not call"),
        "fallback": ("This flags a low or horizontal body position. It is not an injury "
                     "diagnosis."),
    },
}


ACTION_BRANCH = {
    "police_called": "activity",
    "police_not_called": "activity",
    "ambulance_called": "fall",
    "ambulance_not_called": "fall",
}


def alerts_for_branch(events, branch):
    """Only observations addressed by this branch's decision."""
    return [event for event in events if ALERT_BRANCH.get(event.get("category")) == branch]


def pending_alerts(events, escalations):
    """A decision settles only its own branch; deferred rows settle neither.

    Older mixed-branch logs stay immutable. Their police decisions cannot settle a
    medical observation (or vice versa), even when both IDs were stored in one row.
    """
    decided = {(identifier, ACTION_BRANCH[record["action"]])
               for record in escalations if record["action"] in ACTION_BRANCH
               for identifier in record["event_ids"]}
    return [event for event in events
            if event.get("category") in ALERT_BRANCH
            and (event.get("event_id"), ALERT_BRANCH[event["category"]]) not in decided]


def primary_branch(alerts):
    """Which prompt to raise first.

    When both branches are waiting the police prompt is primary, as specified. The medical
    action is still offered on the same card: a possible person down must not disappear
    behind the other branch's question.
    """
    branches = {ALERT_BRANCH[event["category"]] for event in alerts}
    if "activity" in branches:
        return "activity"
    return "fall" if "fall" in branches else None


def secondary_fall_alerts(alerts, branch):
    """Person-down observations waiting behind a primary police prompt."""
    if branch != "activity":
        return []
    return [event for event in alerts if ALERT_BRANCH[event["category"]] == "fall"]
