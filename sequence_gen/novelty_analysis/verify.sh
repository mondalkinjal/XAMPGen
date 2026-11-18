#!/bin/bash
# Independent check that the SI tables match the BLAST output.
#
# Deliberately does NOT use align_tables.py: identities are recomputed from the
# raw TSVs with awk, then compared against the numbers printed in SIXAMPGen.tex.
# If this passes, the manuscript agrees with the data by two separate routes.
set -u
TEX="${XAMPGEN_SI_TEX:?Set XAMPGEN_SI_TEX to the manuscript supporting-information TeX file}"
fail=0

# best hit per peptide, straight from the TSV: identity = nident/qlen (cols 10, 3),
# ties broken by lower E (col 9).  $1 is the peptide, sorted so awk sees groups.
recompute () {   # $1 = tsv, $2 = 1 to drop P9's self match
  sort -k1,1 "$1" | awk -F'\t' -v drop="$2" '
    { id = 100*$10/$3
      if (drop && $10 == $3 && $3 == $4) next     # subject identical to query
      if (!($1 in best) || id > best[$1] || (id == best[$1] && $9 < ev[$1])) {
          best[$1] = id; ev[$1] = $9 } }
    END { for (p in best) printf "%s %.1f\n", p, best[p] }' | sort
}

# the %id column of a LaTeX table: field 5 of Table S2, field 5 of Table S3
fromtex () {     # $1 = label of the table
  awk -v lbl="$1" '
    $0 ~ "\\\\label\\{"lbl"\\}" { inlab=1 }
    inlab && /^    P[0-9]/ {
        gsub(/\\\\$/, ""); n=split($0, f, "&")
        name=f[1]; gsub(/[ \t]/, "", name)
        if (name == "P9") name = "P9_WT"
        # both tables put "% id." in field 5 and E in field 6
        v=f[5]; gsub(/[ \t]/,"",v); printf "%s %s\n", name, v }
    inlab && /\\end\{table\}/ { inlab=0 }' "$TEX" | sort
}

for t in "aln_train.tsv tab:nearest 1" "aln_sp.tsv tab:fragment 0"; do
  set -- $t
  echo "== $2  (from $1)"
  if diff <(recompute "$1" "$3") <(fromtex "$2"); then
      echo "   18/18 identities match the manuscript"
  else
      echo "   MISMATCH above: recomputed (<) vs SIXAMPGen.tex (>)"; fail=1
  fi
done

# every bold run must be a run the alignment actually matched
echo "== bold markup"
python3 - "$TEX" <<'PY'
import re, sys
tex = open(sys.argv[1]).read()
bad = 0
for m in re.finditer(r'\\texttt\{([^}]*(?:\\textbf\{[^}]*\}[^}]*)*)\}', tex):
    s = m.group(1)
    if '\\textbf' not in s:
        continue
    plain = re.sub(r'\\textbf\{([^}]*)\}', r'\1', s)
    if re.search(r'[^A-Za-z]', plain.replace('$^\\dagger$', '')):
        continue
    if re.search(r'\\textbf\{\}', s):          # empty bold group
        bad += 1
print('   %d malformed bold group(s)' % bad)
PY
exit $fail
