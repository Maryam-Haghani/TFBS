# conda install -c bioconda bedtools
# !pip install pybedtools # pybedtools is a wrapper around the BEDTools binary
from pybedtools import BedTool

base_dir = "../inputs/fastas/"
genome_path = f"{base_dir}/Arabidopsis_thaliana.TAIR10.dna_sm.toplevel.fa"
gene_name= "AT2G21510"
position = "2: 9,210,623-9,213,102:-1" # length = 2480
# promoter+200: Chr2:9,212,902–9,214,102 (strand −)

def get_start_end_strand(position):
    # Get 1000 bp + TSS + 200bp:
    chr, start_end, direction = position.replace(" ", "").split(":")
    start, end = start_end.replace(",", "").split("-")
    start, end = int(start), int(end)

    if direction == "-1":
        strand = "-"
        TSS = end
        end_bed = TSS + 1000
        start_bed = TSS - 200
    else:
        strand = "+"
        TSS = start
        start_bed = TSS - 1000
        end_bed = TSS + 200

    return (chr,
            start_bed-1, # 0-based start
            end_bed,     # 1-based exclusive end
            strand)


chr, start_bed, end_bed, strand = get_start_end_strand(position)
print(f"start_bed: {start_bed}, end_bed: {end_bed}")


# build a one-line BED entry from string
name = f"{gene_name}_promoter_TSS_plus200"
bed = BedTool(f"{chr}\t{start_bed}\t{end_bed}\t{name}\t0\t{strand}\n", from_string=True)
# calls bedtools getfasta
bed.sequence(
    fi=genome_path,
    s=False,             # s=True: respects the strand in BED
    name=True,
    fo=f"{base_dir}/{gene_name}.fa"
)

print(f"Wrote FASTA to {gene_name}.fa")