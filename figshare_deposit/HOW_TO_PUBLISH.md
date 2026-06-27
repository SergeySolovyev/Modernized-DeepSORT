# How to publish on Figshare (you do the final steps)

I prepared the file and metadata; the upload and the "Publish" click are yours, since
publishing requires logging in with your own Figshare credentials.

## Files in this folder
- `Solovev_Precision-at-the-Gate_2026.pdf` — the paper (8 pp) — this is what you upload.
- `DESCRIPTION.md` — ready-to-paste title / abstract / keywords / license / category.

## Steps (web, ~3 minutes)
1. Go to https://figshare.com and sign in (or sign in with your institutional / ORCID login).
   Many universities, incl. HSE, have a Figshare presence — a personal figshare.com account works too.
2. Click **Upload** (top right) and drag in `Solovev_Precision-at-the-Gate_2026.pdf`.
3. On the item form, fill the fields from `DESCRIPTION.md`:
   - Title, Description (abstract), Authors, Categories, Keywords.
   - **Item type:** Preprint.
   - **License:** CC BY 4.0 (or your choice).
   - Add the GitHub link under References / Related materials.
4. **Save changes** to keep it as a private draft and review the preview.
5. When you are happy, click **Publish**. Figshare then mints a permanent **DOI** and the item
   becomes public.

## Before you publish — two things to know
- **It is public and permanent.** A published Figshare item gets a DOI and can be *retracted/
  versioned* but **not fully deleted**. You *can* upload new versions later (the DOI stays the same).
- **Preprint vs. your SCOPUS plan.** This deposit is a public preprint. Most journals (Elsevier,
  Springer, IEEE, MDPI) allow posting a preprint before submission, and a Figshare DOI timestamps
  your priority — but a few venues have specific preprint policies, so check your target journal's
  rule first if you have one in mind. If you would rather wait until after journal submission, hold
  off on step 5 (you can keep it as a private draft).

## Optional extras I can add to the deposit (just ask)
- The gating-study data (`paper/experiments/gating_study.csv`) as a supplementary file.
- The popular-science slides (`Точность_на_входе.pdf`) as related media.
- A one-command Figshare-API upload script (you run it locally with your own token in an
  environment variable — I never see the token).
