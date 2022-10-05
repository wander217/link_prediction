from dataset import DocDataset, DocAlphabet

alphabet = DocAlphabet(path=r'F:\project\python\link_prediction\asset\alphabet.txt', max_len=100)
dataset = DocDataset(path=r'F:\project\python\doc_gen\dataset\train.json', alphabet=alphabet, knn_num=3)

data = dataset.__getitem__(0)
(graph, labels, texts, lengths, bboxes, masks) = data
# print(texts[0])
# print(masks[0])
src, dst = graph.edges()
# print(src)
# print(dst)
# print(labels)
# for i in range(len(src)):
#     if src[i] == 9:
#         print(src[i], dst[i])
for i in range(len(labels)):
    if labels[i]:
        print(src[i], dst[i])
