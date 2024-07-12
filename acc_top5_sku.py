import os

dataset_dir = "../datasets/output/sku100k/top_5" # Lightglue v2
# dataset_dir = "/home/gordian/Desktop/alexander/examples/metatrained_benchmarks/sku100_v2/sku100_v2_/changed_dims_400"
# dataset_dir = "/home/gordian/Desktop/alexander/examples/metatrained_benchmarks/sku100_v2/output_top_5" # DEEPEMD v2
# dataset_dir = "/home/gordian/Desktop/alexander/examples/metatrained_benchmarks/SKU100/output_top_5" # DEEPEMD v1

# dataset_dir = "/home/gordian/Desktop/sku100_v2/output_top_5" # Lightglue v2

total_samples = len(os.listdir(dataset_dir))

true_positive = 0
for folder in os.listdir(dataset_dir):
    imgs = os.listdir(os.path.join(dataset_dir, folder))
    query_img = f"query_img_{folder}"
    imgs.remove(query_img)
    print("query images: ", ".".join(folder.split(".")[:-1]))
    print(".".join(folder.split(".")[:-1]))
    found = False
    for img in imgs:
        print(".".join(img.split(".")[:-1]))
        if ".".join(img.split(".")[:-1]) == ".".join(folder.split(".")[:-1]):
            true_positive += 1
            found = True
            break
    # if not found:
    #     print(folder)

accuracy = (true_positive/total_samples) * 100
print("Top 5 accuracy: ", accuracy)

# lightglue v2
# Top 5 accuracy:  84.0958605664488

# lightglue v1
# Top 5 accuracy:  37.28813559322034

# DEEPEMD pretrained v1
# Top 5 accuracy:  76.66666666666667

# DEEPEMD metatrained v1
# Top 5 accuracy:  42.3728813559322

# DEEPEMD metatrained v2
# Top 5 accuracy:  31.154684095860567