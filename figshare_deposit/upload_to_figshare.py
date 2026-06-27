#!/usr/bin/env python3
"""Upload the "Precision at the Gate" preprint to Figshare as a PRIVATE draft.

YOU run this with YOUR own token. The token is read from an environment variable,
so it never appears in the script and is never shared:

    Windows PowerShell:
        $env:FIGSHARE_TOKEN = "PASTE_YOUR_TOKEN"
        py -3.14 upload_to_figshare.py

    bash:
        FIGSHARE_TOKEN=PASTE_YOUR_TOKEN python3 upload_to_figshare.py

How to get a token (one time):
    figshare.com -> your avatar -> "Applications" -> "Create Personal Token"
    -> give it a name -> copy the token (you see it once).

What this does:
    1. creates a PRIVATE draft article with title/abstract/keywords,
    2. uploads Solovev_Precision-at-the-Gate_2026.pdf to it.
    It does NOT publish. Open the printed link, set Category + License, review,
    then click Publish yourself (publishing is public + permanent, so review first).

No third-party packages required (standard library only).
"""
import os, sys, json, hashlib, urllib.request, urllib.error

BASE = "https://api.figshare.com/v2"
TOKEN = os.environ.get("FIGSHARE_TOKEN", "").strip()
HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, "Solovev_Precision-at-the-Gate_2026.pdf")

DESCRIPTION = (
    "Modernizing a 2017-era DeepSORT tracker by swapping its precomputed detections and fixed "
    "appearance encoder for a contemporary detector (YOLOv8m) and re-identification model (OSNet) "
    "raises mean HOTA on six MOT-Challenge videos from 40.17 to 52.63. This headline gain is "
    "unsurprising; the contribution of this paper is the audit underneath it. The gain is "
    "detection-dominated: on the MOT16 split the detection term rises by +12.30 HOTA against only "
    "+6.25 for association, and under ground-truth boxes a generic ImageNet backbone reaches 89.2 "
    "HOTA versus 89.9 for a dedicated OSNet. The mean also hides a regression: at its default gate "
    "the modernized tracker scores below the 2017 baseline on the least-precise clip (TUD-Campus, "
    "35.14 vs 39.86 HOTA). Across all six videos the lowest detector-confidence gate scores no "
    "higher than the highest, and on the low-precision clips lowering the gate sharply lowers HOTA, "
    "with the size of the penalty tracking per-clip detector precision (Pearson r = -0.91, n = 6; "
    "reported as a hypothesis-generating association, not a calibrated law). The cause is specific "
    "and reproducible: the modern live pipeline drops the original DeepSORT confidence and "
    "non-maximum-suppression admission gate, so ungated false positives spawn spurious tracks. A "
    "gating-restoration ablation recovers the loss, and raising the per-clip confidence threshold "
    "lifts the worst sequence from a 4.72-point deficit to +6.43 over baseline. We frame the result "
    "as an audit rather than a new method: modern SORT-family trackers already instantiate this "
    "admission gate as a tuned hyperparameter, and a routine modernization can silently drop it. "
    "All detector results use a pinned configuration (ultralytics 8.4.79, yolov8m.pt, input size "
    "1280). Code, data, and reproduction scripts: "
    "https://github.com/SergeySolovyev/Modernized-DeepSORT"
)

META = {
    "title": "Precision at the Gate: Per-Clip Detector Confidence Outranks Recall in a Modernized DeepSORT Pipeline",
    "description": DESCRIPTION,
    "keywords": ["multi-object tracking", "tracking-by-detection", "DeepSORT", "HOTA",
                 "detector confidence", "non-maximum suppression", "gating",
                 "person re-identification", "MOT-Challenge", "YOLOv8", "OSNet"],
    "references": ["https://github.com/SergeySolovyev/Modernized-DeepSORT"],
    "defined_type": "preprint",
}


def req(method, url, data=None, binary=None, auth=True):
    headers = {}
    if auth:
        headers["Authorization"] = "token " + TOKEN
    body = None
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif binary is not None:
        body = binary
    r = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        sys.exit("Figshare API error %s on %s %s:\n%s" % (e.code, method, url, e.read().decode("utf-8", "replace")[:600]))


def main():
    if not TOKEN:
        sys.exit("Set FIGSHARE_TOKEN first (see the header of this file).")
    if not os.path.exists(PDF):
        sys.exit("PDF not found next to the script: " + PDF)

    # 1) create the private draft article
    _, loc = req("POST", BASE + "/account/articles", data=META)
    art_url = loc.get("location") or (BASE + "/account/articles/" + str(loc.get("entity_id")))
    art_id = art_url.rstrip("/").split("/")[-1]
    print("[1/4] created private draft article", art_id)

    # 2) register the file (name, md5, size)
    blob = open(PDF, "rb").read()
    md5 = hashlib.md5(blob).hexdigest()
    _, floc = req("POST", art_url + "/files",
                  data={"name": os.path.basename(PDF), "md5": md5, "size": len(blob)})
    file_url = floc["location"]
    print("[2/4] file slot registered (%d bytes)" % len(blob))

    # 3) upload the file in parts to the storage service
    _, finfo = req("GET", file_url)
    upload_url = finfo["upload_url"]
    _, info = req("GET", upload_url, auth=False)
    parts = info["parts"]
    for p in parts:
        chunk = blob[p["startOffset"]:p["endOffset"] + 1]
        req("PUT", "%s/%d" % (upload_url, p["partNo"]), binary=chunk, auth=False)
        print("      uploaded part %d/%d" % (p["partNo"], len(parts)))
    # mark the upload complete
    req("POST", file_url)
    print("[3/4] upload complete")

    # 4) done -- leave it as a private draft for you to review + publish
    print("[4/4] DRAFT READY (private, not yet public).")
    print("\nOpen it, set Category + License, review, then click Publish:")
    print("   https://figshare.com/account/articles/" + art_id)


if __name__ == "__main__":
    main()
