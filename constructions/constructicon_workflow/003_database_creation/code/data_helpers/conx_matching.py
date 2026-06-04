#!/usr/bin/env python3

# Finding construction matches from syntax tree of sentence


def match_feats(word, feats: dict) -> bool:
    if feats.get("deprel") and word.deprel != feats["deprel"]:
        return False
    if feats.get("POS") and word.upostag != feats["POS"]:
        return False
    if feats.get("form") and feats["form"] not in word.morph_analysis.form:
        return False
    if feats.get("instance") and word.text != feats["instance"]:
        return False
    return True


def check_order(anchor, chain, order: str) -> bool:
    if order is None:
        return True
    
    other_nodes = chain[1:]  # exclude anchor

    if order == "BEFORE":
        return all(anchor.id < w.id for w in other_nodes)

    if order == "AFTER":
        return all(anchor.id > w.id for w in other_nodes)

    return True


def get_chain(stanza_words: list, direction: str, length: int, anchor_feats: dict, members_feats: list[dict]):
    """
    Määratud suunaga.
    direction: "UP"/"DOWN"
    anchor_feats: {"deprel": str | None, "POS": str | None, "form": str | None, "instance": str | None, "order": "BEFORE" | "AFTER" | None}
    members_feats: [{"deprel": str | None, "POS": str | None, "form": str | None, "instance": str | None, "order": "BEFORE" | "AFTER" | None}, {"deprel": str | None, "POS": str | None, "form": str | None, "instance": str | None, "order": "BEFORE" | "AFTER" | None}, ...]
    """
    #lookup
    id2word = {w.id: w for w in stanza_words}
    children = {}

    for w in stanza_words:
        children.setdefault(w.head, []).append(w)

    conx_ids = []
    conx_headids = []

    anchors = [w for w in stanza_words if match_feats(w, anchor_feats)]
   
    for anchor in anchors:
        current_chain = [anchor]
        current_ids = [anchor.id]
        current_headids = [anchor.head]

        current = anchor
        success = True

        for step in range(length-1):
            expected_feats = members_feats[step]

            next_word = None

            if direction == "UP":
                if current.head == 0:
                    success = False
                    break
                candidate = id2word.get(current.head)
                if candidate and match_feats(candidate, expected_feats):
                    next_word = candidate

            elif direction == "DOWN":
                for child in children.get(current.id, []):
                    if match_feats(child, expected_feats):
                        next_word = child
                        break
           
            if not next_word:
                success = False
                break

            current_chain.append(next_word)
            current_ids.append(next_word.id)
            current_headids.append(next_word.head)
            current = next_word
        
        if success: # after finding a match, we will check the order of chain members
            order = anchor_feats.get("order")

            if check_order(anchor, current_chain, order):
                conx_ids.append(current_ids)
                conx_headids.append(current_headids)


    return conx_ids, conx_headids

