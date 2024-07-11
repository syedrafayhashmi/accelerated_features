import os

# dataset_dir = "/home/gordian/Desktop/alexander/examples/metatrained_benchmarks/sku100_v2/output_top_1" # DEEPEMD v2
# dataset_dir = "/home/gordian/Desktop/alexander/examples/metatrained_benchmarks/SKU100/output_top_1" # DEEPEMD v1
#
dataset_dir = "hlcv-project-datasets/output/sku100k_v2/top_1" # lightglue v2

total_samples = len(os.listdir(dataset_dir))
true_positive = 0
for folder in os.listdir(dataset_dir):
    imgs = os.listdir(os.path.join(dataset_dir, folder))
    query_img = f"query_img_{folder}"
    imgs.remove(query_img)
    if ".".join(imgs[0].split(".")[:-1]) == ".".join(folder.split(".")[:-1]):
        true_positive += 1
    # if not found:
    #     print(folder)

accuracy = (true_positive/total_samples) * 100
print("Top 1 accuracy: ", accuracy)


# DEEPEMD metatrained v2
# Top 1 accuracy:  18.954248366013072
# Top 5 accuracy:  31.154684095860567

# DeepEMD metatrained v1
# Top 1 accuracy:  18.64406779661017
# Top 5 accuracy:  42.3728813559322

# DeepEMD pretrained v1
# Top 1 accuracy:  11.864406779661017
# Top 5 accuracy:  37.28813559322034

# lightglue v1
# Top 1 accuracy:  51.66666666666667
# Top 5 accuracy:  76.66666666666667

# lightglue v2
# Top 1 accuracy:  64.70588235294117
# Top 5 accuracy:  84.0958605664488