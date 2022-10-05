from dataset import DocLoader, DocAlphabet

alphabet = DocAlphabet(path=r'F:\project\python\link_prediction\asset\alphabet.txt', max_len=100)
loader = DocLoader(
    num_workers=0,
    batch_size=4,
    drop_last=True,
    shuffle=True,
    pin_memory=False,
    alphabet=alphabet,
    dataset={
        "path": r'F:\project\python\doc_gen\dataset\train.json',
        "knn_num": 6
    }
).build()

for (graphs, labels,
     texts, bboxes,
     masks, node_factors,
     node_sizes) in loader:
    print(labels.size())
    print(texts.size())
    print(bboxes.size())
    print(node_factors.size())
    print(node_sizes)
    break
