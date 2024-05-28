from gtsrb import GtsrbModule


def main():
    gtsrb_module = GtsrbModule(data_path="/media/farnez/Data/DATASETS/GTSRB-normal/")
    gtsrb_module.prepare_data()
    print("Done!")

if __name__ == "__main__":
    main()