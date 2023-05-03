
import os
import pandas as pd
from termcolor import colored
model1_dir = '/home/ngzhili/uoais/output/R50_rgbdconcat_mlc_occatmask_hom_concat_syntable'
model2_dir = '/home/ngzhili/uoais/output/R50_rgbdconcat_mlc_occatmask_hom_concat_uoais_sim'

csv_type = 'no_cg_net_osd_results.csv' #'osd_results.csv' #'cg_net_osd_results.csv'
all_files = os.listdir(model1_dir)    
model1_csv_files = list(filter(lambda f: f.endswith(csv_type), all_files))[0]
all_files = os.listdir(model2_dir)    
model2_csv_files = list(filter(lambda f: f.endswith(csv_type), all_files))[0]


df1 = pd.read_csv(os.path.join(model1_dir,model1_csv_files))
df2 = pd.read_csv(os.path.join(model2_dir,model2_csv_files))
df1 = df1.iloc[0]
df2 = df2.iloc[0]
df_name1 = "syntable"
df_name2 = "uoais_sim"
df_list = [{df_name1:df1},{df_name2:df2}]
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
eval_type_list = ["Visible","Amodal","Occlusion","Occlusion Classification"]
for eval_type in eval_type_list:


    if eval_type != "Occlusion Classification":
        print(colored(f"{eval_type} Metrics for OSD", "green", attrs=["bold"]))
        print(colored("---------------------------------------------", "green"))

        for df_dict in df_list:
            
            for key,item in df_dict.items():
                df_name = key
                df = item

                print(colored(f"                    {df_name}","blue", attrs=["bold"]))
                print("    Overlap    |    Boundary")
                print("  P    R    \033[4mF\033[0m  |   P    R    \033[4mF\033[0m  |  \033[4m%75\033[0m | mIoU")
                print("{:.1f} {:.1f} {:.1f} | {:.1f} {:.1f} {:.1f} | {:.1f} | {:.4f}".format(
                    df[f'{eval_type} Overlap P'], df[f'{eval_type} Overlap R'], 
                    df[f'{eval_type} Overlap F'],
                    df[f'{eval_type} Boundary P'], df[f'{eval_type} Boundary R'], 
                    df[f'{eval_type} Boundary F'],
                    df[f'{eval_type} %75'], df[f'{eval_type} mIoU']
                ))
                print(colored("---------------------------------------------", "green"))
    else:
        print(colored("Occlusion Classification on OSD", "green", attrs=["bold"]))
        print(colored("---------------------------------------------", "green"))
        for df_dict in df_list:
            
            for key,item in df_dict.items():
                df_name = key
                df = item
                print(colored(f"     {df_name}","blue", attrs=["bold"]))
                print("  P   R   \033[4mF\033[0m   \033[4mACC\033[0m")
                print("{:.1f} {:.1f} {:.1f} {:.1f}".format(
                    df[f'{eval_type} P'], df[f'{eval_type} R'], df[f'{eval_type} F'], df[f'{eval_type} Accuracy']        
                ))
                print(colored("---------------------------------------------", "green"))
