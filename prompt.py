import pickle
import re

def get_prompt(shape_path):
    info_path = shape_path[:-4] + "_info.pkl"
    with open(info_path, "rb") as f:
        info = pickle.load(f)
        name = info["name"]
        dataset = info["org_dataset"]
    if dataset in ["dt4d_human", "faust", "scape", "topkids", "kids"]:
        return "human"
    if dataset == "tosca":
        matches = re.search(r"^(.*?)(\d+)$", name)
        shape_name = matches.group(1)
        if shape_name == "michael" or shape_name == "victoria" or shape_name == "david":
            return "human"
        return shape_name
    if dataset == "shrec20":
        matches = re.search(r"^(.*?)(_.+)?$", name)
        shape_name = matches.group(1)
        return shape_name
    if dataset == "dt4d_animal":
        matches = re.search(r"^(.*?)(([A-Z]|\d).*)?$", name)
        shape_name = matches.group(1)
        return shape_name
    if dataset == "smal":
        category, name = name.split("/")
        if category == "big_cats":
            if name == "muybridge_107_110_03":
                return "cat"
            if "lion" in name or "Lion" in name:
                return "lion"
            if name == "muybridge_132_133_07" or name == "00211799_ferrari":
                return "tiger"
            if name == "cougar":
                return "cougar"
        if category == "dogs":
            if "fox" in name:
                return "fox"
            if name == "dog_alph" or name == "dog2" or name == "NORTHERN-INUIT-DOG-3" or "wolf" in name:
                return "wolf"
            return "dog"
        if category == "horses":
            return "horse"
        if category == "cows":
            return "cow"
        if category == "hippos":
            return "hippo"
    if dataset == "densecorr3d":
        for i in range(5):
            print("Warning: Assuming densecorr3d has only cars")
        return "car"
    if dataset == "shrec07":
        print(name)
        return name
            
    raise NotImplementedError()