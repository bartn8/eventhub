SCANNET_HEIGHT = 480
SCANNET_WIDTH = 640

DATA_SPLIT = {'none': [],}

_VALIDATION_SEQS = []
_TRAIN_SEQS = ['4aef651da7', '38fcf02d0b', 'e0de253456', '72f527a47c', '8fc40ba77b', '2a496183e1', '54b005d19d', '3c8d535d49', 'd7abfc4b17', 'e3ad7115db', '635852d56e', '4c5c60fa76', '75d29d69b8', 'd2f44bf242', '0e350246d3', 'c9a8357e8f', 'bf07750a0b', '82ff39b7ef', 'b97261909e', '48573f4c95', '413085a827', '1a3100752b', '9dc5ad040f', 'dfa70fb232', 'bb05a0c48c', '9471b8d485', 'e3b3b0d0c7', '0caa1ae59a', '4422722c49', 'deb1867829', '281bc17764', 'e8e81396b6', '10c8ab99f4', '55b2bf8036', '785e7504b9', 'b3ac0beef0', '320c3af000', '9bfbc75700', '61adeff7d5', 'c47168fab2', '95748dd597', 'e3ef8b690b', '39e6ee46df', '1730c7d709', 'fd8560cfd6', '98b4ec142f', 'c40466a844', '6da1d5ab04', '7b4a316aea', '3423e509af', '2b71155e0d', '46638cfd0f', '9c7b4394af', '5a9cdde1ba', 'ccfd3ed9c7', '1eacc65607', '25bde9e167', 'e69064f2f3', '67d702f2e8', '2f5996ff01', 'faba6e97d7', '7c31a42404', '303745abc7', '46001f434d', '6183f0657d', 'adf4ab4a53', '6248c6742d', '2779f8f9e2', '4423a61d09', 'acd69a1746', '1117299565', 'cd0b6082d2', '355e5e32db', 'a8f7f66985', '7e7d2e8640', '3ce6d36ab5', '88627b561e', 'c856c41c99', '8f82c394d6', '5ea3e738c3', 'fb9b4c2f15', '80ffca8a48', 'f248c2bcdc', '5334a4164a', '484ad681df', 'bf50f418ba', '58f6a5c5ec', '37562e7f48', 'dd685be466', 'd1345a65c1', '7f68c514bd', '04d0dc245b', 'fb152519ad', '9084d4cd97', 'f8d5147d1d', '2970e95b65', '709ab5bffe', 'c465f388d1', '73f9370962', '2634683a9f', 'a4d48ea6b3', '95d525fbfd', 'b2632b738a', 'b09431c547', 'c8f2218ee2', '4897e95232', '4808c4a397', 'eb8ef9b4cc', '9444b90aaa', '37c9538a2b', '6126572846', '3b90310b1c', 'c0da8f4a4d', 'aa852f7871', '1204e08f17', '4d451d9c36', '97e5512e91', '3799bd47b3', '12c0f7a7da', '4380e4646a', 'ab6983ae6c', '06b5863f73', '0c5385e84b', '53755e535e', 'c2d714d386', 'f8eac0ad24', '29c7afafed', '1d003b07bd', '2a1b555966', 'a30646cae6', '511061232', '7d8d37ca38', '5d902f1593', '1bb93d185e', 'd054227009', '364f01bc18', '246fe09e98', 'c601466b77', '5a269ba6fe', 'c8eeef6427', 'e909f8213d', '7c0ba828a9', '639f2c4d5a', '523657b4d0', '612f70fe00', '5654092cc2', '07f5b601ee', '210f741378', '302a7f6b67', '77b40ce601', '06bc6d1b24', '867d97cf3d', 'abf29d2474', 'ec2cb8dae1', '4ea827f5a1', 'bc2fce1d81', '59e3f1ea37', 'b0fe0c610f', '6bd39ac392', '58960ff105', 'a08d9a2476', '2489b7f4fe', 'd7b871aaa8', '20871b98f3', '124a6e789b', '5aeac3800a', '3aa115e55e', 'd986399f4c', 'd6a77f7c22', 'f25f5e6f63', '13285009a4', 'e3ecd49e2b', 'a9e4791c7e', 'dec0b11090', 'c07c707449', '48701abb21', '1b9692f0c7', 'd807fb583b', '9d8fcc4215', '88f265fe25', '25bae29ab3', '5bc6227191', '85dc2702b7', '85251de7d1', 'e7ccd75e5d', '3e7e4b07c4', '8013901416', '43cd995c51', 'e2caaaf5b5', 'e4007ff6b5', '036bce3393', '260db9cf5a', '16c9bd2e1e', 'd6bb698875', 'defd3457db', '260fa55d50', 'b0f057c684', 'daffc70503', 'ca0c580422', '99010a8938', 'eab5494dca', '27dc178a3d', 'b08a908f0f', '8890d0a267', '442b144761', '079a326597', 'f6659a3107', '617326da3e', '4c141d5b1b', 'f97de2c3e9', '64672b5bf5', '70f0e494b2', 'd61691f945', 'd537ef1d41', '00777c41d4', '69e5939669', 'c4aaedcfd1', '13b4efaf62', 'eeeb9836b8', 'fb564c935d', '3391ff8a71', '652d9cb0d7', 'c842edbdf5', '666d04a14a', 'eaa6c90310', 'be05b26a38', 'ed2216380b', '2748de13fb', '696317583f', '7b04052ad0', '1c7a683c92', '30f4a2b44d', 'd1f82299d0', 'd240136ce4', '3e928dc2f6', 'ff17657f71', '1c4b893630', 'ef18cf0708', '6ad6cef000', '9cfea269dd', '8737a0d1ad', 'b1d75ecd55', '909a9ea5fc', 'c0f5742640', 'e3c1da58dd', '0f3474b837', '45d2e33be1', 'ab11145646', '66c98f4a9b', '6464461276', '04df8734b7', '81a82c3618', '251443268c', '132cb783ed', 'c31ebd4b22', '712b9ae775', 'a1d9da703c', 'b20a261fdf', '871efc90fa', '51bdbf173f', 'e050c15a8d', '15c4aa5bbb', '9ef5fc6271', '480ddaadc0', 'c026d108e0', '324d07a5b3', 'a31b2ef388', 'b5918e4637', '0452249a1e', '69e56cf0f8', 'cb7785f6ad', 'e4fb2a623b', 'dfac5b38df', 'e8ea9b4da8', 'faec2f0468', '5d152fab1b', '8d0f714398', 'bc400d86e1', '0a5c013435', 'e1aa584dd5', '44c85584ae', '02c2ddee2a', '9816c49e97', '09bced689e', '546292a9db', '4610b2104c', '7fb8ff20e9', 'd918af9c5f', '791a5c253d', '504cf57907', 'a4e227f506', '91fc568d84', '8de35c04a3', '39580e2a43', '82f448db76', 'e9e16b6043', '0271889ec0', 'eea4ad9c04', 'eaab7bcc15', '0d8ead0038', '6b19334aeb', '47b37eb6f9', '618310ed87', 'be0ed6b33c', '3ff873c77e', '7b4cb756d4', '192ab15daf', '3a3745a437', 'ab046f8faf', '09a6767fc2', 'f8e13ab4ae', '6f1848d1e3', '0f25f24a4f', 'aab83fd6f1', '35050f41c5', 'f847086d15', '24b248e676', 'c08d1d52b7', '39f36da05b', '25aa952aa3', 'be8367fcbe', '7c31bccde5', '0658da5bc0', 'e667e09fe6', '1841a0b525', 'f576071590', '1c08823a41', '6b40d1a939', 'c4d4cb61f6', '08bbbdcc3d', 'a492fe77aa', 'dc263dfbf0', '49789448b8', '20ff72df6e', '3d838ee1ab', '0c6c7145ba', 'ce12db9e81', 'cec8312f4e', '70945f435a', '9b74afd2d2', '7739004a45', 'b6d73041c8', '10242d1eaf', '052d72e137', 'b24697b3a1', 'cab239278a', 'f19ca0a52e', '724c40236c', '0f0191b10b', 'e81c8b3eec', 'ad2d07fd11', 'e4e625a3e4', '589f5c7c58', 'fe1733741f', '5c215ef3b0', 'a892730b61', '94b1acde81', '076c822ecc', '1a8e0d78c0', 'ac250f0ead', 'e5a769dbf5', 'db5293a870', 'bfcfe53c6a', '0c7962bd64', '8be0cd3817', 'bac7ee3b1b', 'a23f391ba9', '56a0ec536c', '2f6f83ea1f', 'ef25276c25', 'c29b5e479c', '1a130d092a', '8e22c48c20', 'bb0ad8a081', '3caf4324fd', 'aea84db0de', '0e100756bf', 'b0b004c40f', '66ba53719a', '5f0fb991a7', 'cc5ea8026c', '893fb90e89', '56669a70bc', 'de5881aa12', '0e900bcc5c', 'bfd3fd54d2', '41b00feddb', 'f6a9b64a0d', '7f77abce34', '3cbb18c391', 'd551dac194', '7f22d5ef1b', 'f3f016ba3f', 'b4b39438f0', 'b068706ef0', '068ba2946c', 'de3c77cecd', '98fe276aa8', 'f38b0108a1', '4517d988d8', '4318f8bb3c', 'fd361ab85f', '238b940049']

_BROKEN_SEQS = ['8890d0a267', 'c29b5e479c', 'd918af9c5f']
_MISSING_SEQS = []

_REMOVE_SEQS = _BROKEN_SEQS + _MISSING_SEQS

for seq in _REMOVE_SEQS:
    if seq in _VALIDATION_SEQS:
        _VALIDATION_SEQS.remove(seq)
    if seq in _TRAIN_SEQS:
        _TRAIN_SEQS.remove(seq)

#Debug mode
# _VALIDATION_SEQS =_VALIDATION_SEQS[:5]  # Use only the first sequence for validation in debug mode
# _TRAIN_SEQS = _TRAIN_SEQS[:5]  # Use only the first sequence for training in debug mode

_SCAN_LUT = {
    'train': ['param0'],
    'validation': ['param0'],
    'curation': ['param0'],
}

_SCAN_SEQS = {
    'train': _TRAIN_SEQS,
    'validation': _VALIDATION_SEQS,
    'curation': _TRAIN_SEQS,
}

for split in ['train', 'validation', 'curation']:
    DATA_SPLIT[split] = [f'{i:04}/rendered_frames_{mix}/' for i in _SCAN_SEQS[split] for mix in _SCAN_LUT[split]]
