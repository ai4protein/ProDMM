import torch

vocab_list = [
    '<pad>',
    '<cls>',
    '<sep>',
    '<unk>',
    '<bos>',
    '<eos>',
    '<mask>',
    '-',
    '*',
    'L',
    'A',
    'G',
    'V',
    'S',
    'E',
    'R',
    'T',
    'I',
    'D',
    'P',
    'K',
    'Q',
    'N',
    'F',
    'Y',
    'M',
    'H',
    'W',
    'C',
    'aaa', 'aat', 'aac', 'aag', 'ata', 'att', 'atc', 'atg',
    'aca', 'act', 'acc', 'acg', 'aga', 'agt', 'agc', 'agg',
    'taa', 'tat', 'tac', 'tag', 'tta', 'ttt', 'ttc', 'ttg',
    'tca', 'tct', 'tcc', 'tcg', 'tga', 'tgt', 'tgc', 'tgg',
    'caa', 'cat', 'cac', 'cag', 'cta', 'ctt', 'ctc', 'ctg',
    'cca', 'cct', 'ccc', 'ccg', 'cga', 'cgt', 'cgc', 'cgg',
    'gaa', 'gat', 'gac', 'gag', 'gta', 'gtt', 'gtc', 'gtg',
    'gca', 'gct', 'gcc', 'gcg', 'gga', 'ggt', 'ggc', 'ggg',
    'a', 'c', 'g', 't',
    '<aas2cds>',
    '<cds2pre-ncds>',
    '<cds2post-ncds>',
]

TOKEN_TO_IDX = {token: idx for idx, token in enumerate(vocab_list)}
IDX_TO_TOKEN = {idx: token for token, idx in TOKEN_TO_IDX.items()}
vocab_size = len(vocab_list)

# 氨基酸列表（单个字母表示法）
amino_acids = [
    'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I',
    'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C',
    '*', '-',  # 包含特殊氨基酸（如终止符 '*'，缺失 '-'）
]

# 密码子列表（所有可能的三联体）
codons = [
    'aaa', 'aat', 'aac', 'aag', 'ata', 'att', 'atc', 'atg',
    'aca', 'act', 'acc', 'acg', 'aga', 'agt', 'agc', 'agg',
    'taa', 'tat', 'tac', 'tag', 'tta', 'ttt', 'ttc', 'ttg',
    'tca', 'tct', 'tcc', 'tcg', 'tga', 'tgt', 'tgc', 'tgg',
    'caa', 'cat', 'cac', 'cag', 'cta', 'ctt', 'ctc', 'ctg',
    'cca', 'cct', 'ccc', 'ccg', 'cga', 'cgt', 'cgc', 'cgg',
    'gaa', 'gat', 'gac', 'gag', 'gta', 'gtt', 'gtc', 'gtg',
    'gca', 'gct', 'gcc', 'gcg', 'gga', 'ggt', 'ggc', 'ggg',
]

# 获取氨基酸和密码子的索引列表
AA_IDX_LIST = [TOKEN_TO_IDX[aa] for aa in amino_acids if aa in TOKEN_TO_IDX]
CODON_IDX_LIST = [TOKEN_TO_IDX[codon] for codon in codons if codon in TOKEN_TO_IDX]

# 建立氨基酸到密码子的映射
AA_TO_CODONS = {
    'A': ['gct', 'gcc', 'gca', 'gcg'],
    'C': ['tgt', 'tgc'],
    'D': ['gat', 'gac'],
    'E': ['gaa', 'gag'],
    'F': ['ttt', 'ttc'],
    'G': ['ggt', 'ggc', 'gga', 'ggg'],
    'H': ['cat', 'cac'],
    'I': ['att', 'atc', 'ata'],
    'K': ['aaa', 'aag'],
    'L': ['tta', 'ttg', 'ctt', 'ctc', 'cta', 'ctg'],
    'M': ['atg'],
    'N': ['aat', 'aac'],
    'P': ['cct', 'ccc', 'cca', 'ccg'],
    'Q': ['caa', 'cag'],
    'R': ['cgt', 'cgc', 'cga', 'cgg', 'aga', 'agg'],
    'S': ['tct', 'tcc', 'tca', 'tcg', 'agt', 'agc'],
    'T': ['act', 'acc', 'aca', 'acg'],
    'V': ['gtt', 'gtc', 'gta', 'gtg'],
    'W': ['tgg'],
    'Y': ['tat', 'tac'],
    '*': ['taa', 'tag', 'tga'],  # 终止密码子
    # 如果有其他特殊氨基酸，也可以在此添加
}

# 将氨基酸和密码子转换为索引
AA_IDX_TO_CODON_IDXS = {}
for aa, codon_list in AA_TO_CODONS.items():
    aa_idx = TOKEN_TO_IDX.get(aa)
    codon_idxs = [TOKEN_TO_IDX.get(codon) for codon in codon_list if TOKEN_TO_IDX.get(codon) is not None]
    if aa_idx is not None and codon_idxs:
        AA_IDX_TO_CODON_IDXS[aa_idx] = codon_idxs

