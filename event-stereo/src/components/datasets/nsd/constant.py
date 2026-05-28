NSD_HEIGHT = 480
NSD_WIDTH = 640

# NSD_HEIGHT = 412
# NSD_WIDTH = 918

#Motion ablation dataset constants for the NSD dataset.

_MOTION_ABLATION_LUT = {
    'train_h': 'h',
    'train_v': 'v',
    'train_z': 'z',
    'validation_h': 'h',
    'validation_v': 'v',
    'validation_z': 'z',
}

#I have chosen as training sequences for motion ablation the first 40 sequences of the dataset, and the next 10 sequences as validation sequences.

_MOTION_ABLATION_TRAIN_SEQS = {
    'train_h': [i for i in range(40)],
    'train_v': [i for i in range(40)],
    'train_z': [i for i in range(40)],
    'validation_h': [i for i in range(40, 50)],
    'validation_v': [i for i in range(40, 50)],
    'validation_z': [i for i in range(40, 50)],
}

DATA_SPLIT = {'none': [],}

for split in ['train_h', 'train_v', 'train_z', 'validation_h', 'validation_v', 'validation_z']:
    DATA_SPLIT[split] = [f'{i:04}/rendered_frames_{_MOTION_ABLATION_LUT[split]}/' for i in _MOTION_ABLATION_TRAIN_SEQS[split]]

DATA_SPLIT['train_hvz'] = DATA_SPLIT['train_h'] + DATA_SPLIT['train_v'] + DATA_SPLIT['train_z']
DATA_SPLIT['validation_hvz'] = DATA_SPLIT['validation_h'] + DATA_SPLIT['validation_v'] + DATA_SPLIT['validation_z']


# Full dataset constants for the NSD dataset.
# Total sequences: 270
# Selected sequences for validation: 10 (89, 55, 263, 244, 34, 6, 205, 143, 182, 125)
# _VALIDATION_SEQS = [89, 55, 263, 244, 34, 6, 205, 143, 182, 125]
# _VALIDATION_SEQS = []
_VALIDATION_SEQS = [i for i in range(270)]
# Selected sequences for training: 260 (all except the validation sequences)
# _TRAIN_SEQS = [i for i in range(270) if i not in _VALIDATION_SEQS]
_TRAIN_SEQS = [i for i in range(270)]

#BROKEN_SEQS=(0203 0140 0139 0262 0240 0231 0028 0220 0172 0170 0159 0258 0237 0259 0064 0138 0142 0135 0108 0205 0236 0238 0218 0076 0179 0069 0224 0216 0165 0093 0091 0167 0257 0223 0101 0094 0247 0162 0229 0013 0068 0175 0202 0120 0248 0168 0005 0267 0137 0083 0103 0204 0154 0078 0186 0096 0128 0171 0055 0141 0221 0152 0243 0065 0264 0022 0219 0031 0145 0061 0178 0181 0150 0185 0197 0266 0250 0080 0107 0242 0116 0241 0102 0173 0014 0201 0212 0034 0032 0222 0105 0261 0001 0043 0019 0151 0228 0191 0073 0118 0188 0200 0059 0263 0026 0053 0233 0206 0256 0180 0072 0184 0176 0156 0143 0166 0164 0126 0090 0017 0127 0134 0268 0174 0132 0260 0209 0104 0147 0016 0020 0086 0024 0012 0089)
# _BROKEN_SEQS = [203, 140, 139, 262, 240, 231, 28, 220, 172, 170, 159, 258, 237, 259, 64, 138, 142, 135, 108, 205, 236, 238, 218, 76, 179, 69, 224, 216, 165, 93, 91, 167, 257, 223, 101, 94, 247, 162, 229, 13, 68, 175, 202, 120, 248, 168, 5, 267, 137, 83, 103, 204, 154, 78, 186, 96, 128, 171, 55, 141, 221, 152, 243, 65, 264, 22, 219, 31, 145, 61, 178, 181, 150, 185, 197, 266, 250, 80, 107, 242, 116, 241, 102, 173, 14, 201, 212, 34, 32, 222, 105, 261, 1, 43, 19, 151, 228, 191, 73, 118, 188, 200, 59, 263, 26, 53, 233, 206, 256, 180, 72, 184, 176, 156, 143, 166, 164, 126, 90, 17, 127, 134, 268, 174, 132, 260, 209, 104, 147, 16, 20, 86, 24, 12, 89]
_BROKEN_SEQS = []

# _CURATIONS_SEQS = [13, 22, 40, 65, 68, 76, 106, 119, 127, 150, 172, 187, 204, 214, 222, 238, 255, 259]
_CURATIONS_SEQS = []

_DELETE_SEQS = _BROKEN_SEQS + _CURATIONS_SEQS

for seq in _DELETE_SEQS:
    if seq in _VALIDATION_SEQS:
        _VALIDATION_SEQS.remove(seq)
    if seq in _TRAIN_SEQS:
        _TRAIN_SEQS.remove(seq)

#Debug mode
# _VALIDATION_SEQS =_VALIDATION_SEQS[:5]  # Use only the first sequence for validation in debug mode
# _TRAIN_SEQS = _TRAIN_SEQS[:5]  # Use only the first sequence for training in debug mode


_NSD_LUT = {
    'train': ['h', 'v', 'z'],
    # 'validation': ['h', 'h_inv', 'v', 'v_inv', 'z', 'z_inv'],
    'validation': ['z'],
    'curation': ['h', 'v', 'z'],
}

_NSD_SEQS = {
    'train': _TRAIN_SEQS,
    'validation': _VALIDATION_SEQS,
    'curation': _TRAIN_SEQS,
}

for split in ['train', 'validation', 'curation']:
    DATA_SPLIT[split] = [f'{i:04}/rendered_frames_{mix}/' for i in _NSD_SEQS[split] for mix in _NSD_LUT[split]]
