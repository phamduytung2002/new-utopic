import os

bsz = 128
dataset = 'imdb'
n_topic = 100
epochs_1 = 100
epochs_2 = 100
bert_model = 'sentence-transformers/all-MiniLM-L6-v2'

if __name__ == "__main__":
    for stage_1_dist in [0.1, 0.5, 1, 5, 10, 50, 100]:
        print("*******************************")
        print(f'Running {dataset} with stage_1_dist={stage_1_dist}')
        os.system(f'python runs1.py --base-model {bert_model} --dataset {dataset} --n-word 5000 --epochs-1 {epochs_1} --bsz {bsz} --coeff-1-dist {stage_1_dist} --n-cluster {n_topic}')
        
        for stage_2_cons in [0.1, 0.5, 1, 5, 10]:
            for stage_2_dist in [0.1, 0.5, 1, 5, 10]:
                print("*******************")
                print("Running stage 2 with:")
                print(f"stage_2_cons={stage_2_cons}")
                print(f"stage_2_dist={stage_2_dist}")
                os.system(f'python runs23.py --base-model {bert_model} --dataset {dataset} --n-word 5000 --epochs-1 {epochs_1} --epochs-2 {epochs_2} --bsz {bsz} --stage-2-lr 2e-2 --stage-2-repeat 3 --coeff-1-dist {stage_1_dist} --coeff-2-cons {stage_2_cons} --coeff-2-dist {stage_2_dist} --n-cluster {n_topic}')
