#!/bin/bash
# The two BLAST searches behind Tables S2 and S3, with the aligned sequences
# kept (qseq, sseq) so the shared residues can be set in bold.
set -e
if ! command -v blastp >/dev/null 2>&1; then
source /etc/profile.d/modules.sh
module purge; module load blast-plus/2.12.0
fi          # keep this load on its own:
                                                     # it conflicts with python/3.10.10
G="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$G/novelty_analysis"

OPT="-task blastp-short -evalue 200 -max_target_seqs 20"
FMT='6 qseqid sseqid qlen slen qstart qend sstart send evalue nident qseq sseq'

# reference 1: the fine-tuning AMPs, one bare sequence per line -> FASTA -> db
awk '{printf(">amp%d\n%s\n", NR, $0)}' "$G/unique_combined_sequences.txt" > train.fasta
makeblastdb -in train.fasta -dbtype prot -out trainamp >/dev/null
blastp -query "$G/helicity_18.fasta" -db trainamp $OPT -outfmt "$FMT" > aln_train.tsv

# reference 2: UniProtKB/Swiss-Prot; stitle carries the protein name
blastp -query "$G/helicity_18.fasta" -db "$G/piscidin_analysis/swissprot" \
       $OPT -outfmt "$FMT stitle" > aln_sp.tsv

wc -l aln_train.tsv aln_sp.tsv
