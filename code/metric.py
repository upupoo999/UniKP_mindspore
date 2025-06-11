import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['font.family'] = 'STFangsong'
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(8, 6))
sns.set_style("whitegrid")

def get_r2_score():
    r2_scores_list = []
    for i in range(5):
        res = np.array(pd.read_excel(f'./PreKcat_new/{i+1}_all_samples_metrics.xlsx', sheet_name='Sheet1')).T
        Training_test = res[9].tolist()
        Value = res[7].tolist()
        Predict_Label = res[8].tolist()
        Value_test = [Value[i] for i in range(len(Training_test)) if Training_test[i] == 1]
        Predict_Label_test = [Predict_Label[i] for i in range(len(Training_test)) if Training_test[i] == 1]
        Value_test = np.array(Value_test)
        Predict_Label_test = np.array(Predict_Label_test)
        r2_test = r2_score(Value_test, Predict_Label_test)
        r2_scores_list.append(r2_test)
    average_r2 = np.mean(r2_scores_list)
    ax = sns.boxplot(data=[r2_scores_list])
    ax = sns.stripplot(data=[r2_scores_list], color='black', size=6, jitter=True, edgecolor="auto")
    plt.ylabel('R² on test set', fontsize=12)
    plt.xticks([0], ['UniKP'])
    plt.axhline(y=average_r2, color='r', linestyle='--', label=f'平均 R²: {average_r2:.4f}')
    plt.legend()
    plt.ylim(min(r2_scores_list) - 0.05, max(r2_scores_list) + 0.05)
    plt.tight_layout()
    plt.savefig("./metrics/r2_scores_boxplot.png", dpi=300, bbox_inches='tight') 
    plt.show()

def get_rmse_score():
    Value_test = []
    Predict_Label_test = []
    Value_train = []
    Predict_Label_train = []
    res = np.array(pd.read_excel('./PreKcat_new/3_all_samples_metrics.xlsx', sheet_name='Sheet1')).T
    Training_test = res[9].tolist()
    Value = res[7].tolist()
    Predict_Label = res[8].tolist()
    for i in range(len(Training_test)):
        if Training_test[i] == 1:
            Value_test.append(Value[i])
            Predict_Label_test.append(Predict_Label[i])
        else:
            Value_train.append(Value[i])
            Predict_Label_train.append(Predict_Label[i])
    Value_test = np.array(Value_test)
    Predict_Label_test = np.array(Predict_Label_test)
    Value_train = np.array(Value_train)
    Predict_Label_train = np.array(Predict_Label_train)
    rmse_train = np.sqrt(mean_squared_error(Value_train, Predict_Label_train))
    rmse_test = np.sqrt(mean_squared_error(Value_test, Predict_Label_test))
    groups = ['Training set', 'Test set'] 
    unikp_data = [rmse_train, rmse_test]
    x = np.arange(len(groups))  
    width = 0.35                
    fig, ax = plt.subplots(figsize=(6, 4))
    rects = ax.bar(x + width/2, unikp_data, width, 
                label='UniKP', color='#87CEEB')
    ax.set_xticks(x)
    ax.set_xticklabels(groups)  
    ax.set_ylabel('RMSE')        
    ax.legend()     
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom')
    add_labels(rects)
    plt.tight_layout()  
    plt.savefig('./metrics/rmse_histogram.png', dpi=300, bbox_inches='tight')  
    plt.show()
    plt.close(fig)

def get_pcc_scatter_plot():
    res = np.array(pd.read_excel('./PreKcat_new/3_all_samples_metrics.xlsx', sheet_name='Sheet1')).T
    Training_test = res[9].tolist()
    Value = res[7].tolist()
    Predict_Label = res[8].tolist()
    Value_test = [Value[i] for i in range(len(Training_test)) if Training_test[i] == 1]
    Predict_Label_test = [Predict_Label[i] for i in range(len(Training_test)) if Training_test[i] == 1]
    Value_test = np.array(Value_test)
    Predict_Label_test = np.array(Predict_Label_test)
    pcc_test = pearsonr(Value_test, Predict_Label_test)[0]
    n = len(Value_test)
    kde = gaussian_kde([Value_test, Predict_Label_test])
    density = kde([Value_test, Predict_Label_test])
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        Value_test, Predict_Label_test, 
        c=density, 
        cmap='jet', 
        s=10,       
        alpha=0.8   
    )

    divider = make_axes_locatable(ax) 
    cax = divider.append_axes("right", size="5%", pad=0.1)  

    vmin, vmax = density.min(), density.max()
    cbar = fig.colorbar(sc, cax=cax, ticks=np.linspace(vmin, vmax, 6))  
    cbar.set_label('Density', fontsize=12)
    cbar.ax.set_yticklabels([f'{v:.2f}' for v in np.linspace(vmin, vmax, 6)]) 
    ax.text(
        Value_test.min(), Predict_Label_test.max(), 
        f'PCC = {pcc_test:.2f}\nN = {n}', 
        va='top', ha='left', 
        fontsize=12, fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.8, pad=5)
    )

    plt.xlabel(r'$\log_{10}$[experimental $k_{\text{cat}}$ value (s$^{-1}$)]')
    plt.ylabel(r'$\log_{10}$[predicted $k_{\text{cat}}$ value (s$^{-1}$)]')
    plt.tight_layout()
    plt.savefig('./metrics/kcat_correlation_scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def get_rmse_numerical_intervals():
    res = np.array(pd.read_excel('./PreKcat_new/3_all_samples_metrics.xlsx', sheet_name='Sheet1')).T
    Training_test = res[9].tolist()
    Value = res[7].tolist()
    Predict_Label = res[8].tolist()
    Value_5, Predict_Label_5  = [],[]
    Value_4, Predict_Label_4  = [],[]
    Value_3, Predict_Label_3  = [],[]
    Value_2, Predict_Label_2  = [],[]
    Value_1, Predict_Label_1  = [],[]
    Value_0, Predict_Label_0  = [],[]
    Value_neg, Predict_Label_neg  = [],[]
    intervals = ['>5', '5-4', '4-3', '3-2', '2-1', '1-0', '<0']
    for i in range(len(Training_test)):
        if Value[i]>5: 
            Value_5.append(Value[i])
            Predict_Label_5.append(Predict_Label[i])
        elif Value[i]>4 and Value[i]<=5:
            Value_4.append(Value[i])
            Predict_Label_4.append(Predict_Label[i])
        elif Value[i]>3 and Value[i]<=4:
            Value_3.append(Value[i])
            Predict_Label_3.append(Predict_Label[i])
        elif Value[i]>2 and Value[i]<=3:   
            Value_2.append(Value[i])
            Predict_Label_2.append(Predict_Label[i])
        elif Value[i]>1 and Value[i]<=2:
            Value_1.append(Value[i])
            Predict_Label_1.append(Predict_Label[i])
        elif Value[i]>0 and Value[i]<=1:
            Value_0.append(Value[i])
            Predict_Label_0.append(Predict_Label[i])
        else:
            Value_neg.append(Value[i])
            Predict_Label_neg.append(Predict_Label[i])
    rmse_5 = mean_squared_error(Value_5, Predict_Label_5)
    rmse_4 = mean_squared_error(Value_4, Predict_Label_4)
    rmse_3 = mean_squared_error(Value_3, Predict_Label_3)
    rmse_2 = mean_squared_error(Value_2, Predict_Label_2)
    rmse_1 = mean_squared_error(Value_1, Predict_Label_1)
    rmse_0 = mean_squared_error(Value_0, Predict_Label_0)
    rmse_neg = mean_squared_error(Value_neg, Predict_Label_neg)
    unikp_rmse = [rmse_5, rmse_4, rmse_3, rmse_2, rmse_1, rmse_0, rmse_neg]
    x = np.arange(len(intervals))  
    width = 0.35                   
    fig, ax = plt.subplots(figsize=(8, 5))

    rects_unikp = ax.bar(
        x, unikp_rmse, 
        width, 
        label='UniKP', 
        color='#87CEEB'  
    )
    ax.set_xticks(x)
    ax.set_xticklabels(intervals)  
    ax.set_ylabel('RMSE', fontsize=12)        
    ax.set_xlabel(r'$\log_{10}$[experimental $k_{\text{cat}}$ value (s$^{-1}$)]', fontsize=12)
    ax.legend()                  
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f'{height:.4f}',
                xy=(rect.get_x() + rect.get_width()/2, height),
                xytext=(0, 3),  
                textcoords="offset points",
                ha='center', va='bottom'
            )

    add_labels(rects_unikp)
    plt.tight_layout()  
    plt.savefig('./metrics/rmse_numerical_intervals.png', dpi=300, bbox_inches='tight')  
    plt.show()

def get_wildtype_scatter_plot():
    res = np.array(pd.read_excel('./PreKcat_new/3_all_samples_metrics.xlsx', sheet_name='Sheet1')).T
    Training_test = res[9].tolist()
    Value = res[7].tolist()
    Predict_Label = res[8].tolist()
    Type = res[6].tolist()

    Value_wildtype = []
    Predict_Label_wildtype = []
    for i in range(len(Type)):
        if Type[i] == 'wildtype' and Training_test[i] == 1:
            Value_wildtype.append(Value[i])
            Predict_Label_wildtype.append(Predict_Label[i])

    Value_wildtype = np.array(Value_wildtype)
    Predict_Label_wildtype = np.array(Predict_Label_wildtype)
    pcc_test = pearsonr(Value_wildtype, Predict_Label_wildtype)[0]
    n = len(Value_wildtype)
    kde = gaussian_kde([Value_wildtype, Predict_Label_wildtype])
    density = kde([Value_wildtype, Predict_Label_wildtype])
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        Value_wildtype, Predict_Label_wildtype, 
        c=density, 
        cmap='jet', 
        s=10,       
        alpha=0.8   
    )

    divider = make_axes_locatable(ax) 
    cax = divider.append_axes("right", size="5%", pad=0.1)  

    vmin, vmax = density.min(), density.max()
    cbar = fig.colorbar(sc, cax=cax, ticks=np.linspace(vmin, vmax, 6))  
    cbar.set_label('Density', fontsize=12)
    cbar.ax.set_yticklabels([f'{v:.2f}' for v in np.linspace(vmin, vmax, 6)]) 
    ax.text(
        Value_wildtype.min(), Predict_Label_wildtype.max(), 
        f'PCC = {pcc_test:.2f}\nN = {n}', 
        va='top', ha='left', 
        fontsize=12, fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.8, pad=5)
    )

    ax.set_title('Wild-type')
    ax.set_xlabel(r'$\log_{10}$[experimental $k_{\text{cat}}$ value (s$^{-1}$)]')
    ax.set_ylabel(r'$\log_{10}$[predicted $k_{\text{cat}}$ value (s$^{-1}$)]')

    plt.tight_layout()
    plt.savefig('./metrics/wildtype_scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def get_mutant_scatter_plot():
    res = np.array(pd.read_excel('./PreKcat_new/3_all_samples_metrics.xlsx', sheet_name='Sheet1')).T
    Training_test = res[9].tolist()
    Value = res[7].tolist()
    Predict_Label = res[8].tolist()
    Type = res[6].tolist()

    Value_mutant = []
    Predict_Label_mutant = []
    for i in range(len(Type)):
        if Type[i] != 'wildtype' and Training_test[i] == 1:
            Value_mutant.append(Value[i])
            Predict_Label_mutant.append(Predict_Label[i])

    Value_mutant = np.array(Value_mutant)
    Predict_Label_mutant = np.array(Predict_Label_mutant)
    pcc_test = pearsonr(Value_mutant, Predict_Label_mutant)[0]
    n = len(Value_mutant)
    kde = gaussian_kde([Value_mutant, Predict_Label_mutant])
    density = kde([Value_mutant, Predict_Label_mutant])
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        Value_mutant, Predict_Label_mutant, 
        c=density, 
        cmap='jet', 
        s=10,       
        alpha=0.8   
    )

    divider = make_axes_locatable(ax) 
    cax = divider.append_axes("right", size="5%", pad=0.1)  

    vmin, vmax = density.min(), density.max()
    cbar = fig.colorbar(sc, cax=cax, ticks=np.linspace(vmin, vmax, 6))  
    cbar.set_label('Density', fontsize=12)
    cbar.ax.set_yticklabels([f'{v:.2f}' for v in np.linspace(vmin, vmax, 6)]) 
    ax.text(
        Value_mutant.min(), Predict_Label_mutant.max(), 
        f'PCC = {pcc_test:.2f}\nN = {n}', 
        va='top', ha='left', 
        fontsize=12, fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.8, pad=5)
    )

    ax.set_title('Mutant')
    ax.set_xlabel(r'$\log_{10}$[experimental $k_{\text{cat}}$ value (s$^{-1}$)]')
    ax.set_ylabel(r'$\log_{10}$[predicted $k_{\text{cat}}$ value (s$^{-1}$)]')

    plt.tight_layout()
    plt.savefig('./metrics/mutant_scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def get_pcc_wild_mutant():
    Value_wildtype = []
    Predict_Label_wildtype = []
    Value_mutant = []
    Predict_Label_mutant = []
    res = np.array(pd.read_excel('./PreKcat_new/3_all_samples_metrics.xlsx', sheet_name='Sheet1')).T
    Training_test = res[9].tolist()
    Value = res[7].tolist()
    Predict_Label = res[8].tolist()
    Type = res[6].tolist()

    for i in range(len(Type)):
        if Type[i] != 'wildtype' and Training_test[i] == 1:
            Value_mutant.append(Value[i])
            Predict_Label_mutant.append(Predict_Label[i])
        else:
            Value_wildtype.append(Value[i])
            Predict_Label_wildtype.append(Predict_Label[i])
    Value_mutant = np.array(Value_mutant)
    Predict_Label_mutant = np.array(Predict_Label_mutant)
    Value_wildtype = np.array(Value_wildtype)
    Predict_Label_wildtype = np.array(Predict_Label_wildtype)
    pcc_mutant= pearsonr(Value_mutant, Predict_Label_mutant)[0]
    pcc_wildtype = pearsonr(Value_wildtype, Predict_Label_wildtype)[0]
    groups = ['Wild-type', 'Mutant'] 
    unikp_data = [pcc_wildtype, pcc_mutant]
    x = np.arange(len(groups))  
    width = 0.35                
    fig, ax = plt.subplots(figsize=(6, 4))
    rects = ax.bar(x + width/2, unikp_data, width, 
                label='UniKP', color='#87CEEB')
    ax.set_xticks(x)
    ax.set_xticklabels(groups)  
    ax.set_ylabel('PCC')        
    ax.legend()     
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom')
    add_labels(rects)
    plt.tight_layout()  
    plt.savefig('./metrics/pcc_wild_mutant.png', dpi=300, bbox_inches='tight')  
    plt.show()
    plt.close(fig)

def get_pcc_scatter_tem_plot():
    res = np.array(pd.read_excel('./UniKP_mindspore/PreKcat_new/degree_Kcat_5_cv.xlsx', sheet_name='Sheet1')).T
    # Training_test = res[9].tolist()
    Value = res[1].tolist()
    Predict_Label = res[2].tolist()
    Value_test = Value
    Predict_Label_test = Predict_Label
    Value_test = np.array(Value_test)
    Predict_Label_test = np.array(Predict_Label_test)
    pcc_test = pearsonr(Value_test, Predict_Label_test)[0]
    n = len(Value_test)
    kde = gaussian_kde([Value_test, Predict_Label_test])
    density = kde([Value_test, Predict_Label_test])
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        Value_test, Predict_Label_test, 
        c=density, 
        cmap='jet', 
        s=10,       
        alpha=0.8   
    )

    divider = make_axes_locatable(ax) 
    cax = divider.append_axes("right", size="5%", pad=0.1)  

    vmin, vmax = density.min(), density.max()
    cbar = fig.colorbar(sc, cax=cax, ticks=np.linspace(vmin, vmax, 6))  
    cbar.set_label('Density', fontsize=12)
    cbar.ax.set_yticklabels([f'{v:.2f}' for v in np.linspace(vmin, vmax, 6)]) 
    ax.text(
        Value_test.min(), Predict_Label_test.max(), 
        f'PCC = {pcc_test:.2f}\nN = {n}', 
        va='top', ha='left', 
        fontsize=12, fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.8, pad=5)
    )
    ax.text(
        Value_test.max()*0.6, Predict_Label_test.max() * 0.9, 
        'temperature dataset', 
        va='top', ha='left', 
        fontsize=12, fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.8, pad=5)
    )
    # ax.set_xlim(-4, 4)  # 限制x轴范围为0到120
    # ax.set_ylim(-4, 4)   # 限制y轴范围为0到80

    plt.xlabel(r'$\log_{10}$[experimental $k_{\text{cat}}$ value (s$^{-1}$)]')
    plt.ylabel(r'$\log_{10}$[predicted $k_{\text{cat}}$ value (s$^{-1}$)]')
    plt.tight_layout()
    plt.savefig('./UniKP_mindspore/metrics/kcat_correlation_scatter_tem_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def get_pcc_scatter_ph_plot():
    res = np.array(pd.read_excel('./UniKP_mindspore/PreKcat_new/ph_Kcat_5_cv.xlsx', sheet_name='Sheet1')).T
    # Training_test = res[9].tolist()
    Value = res[1].tolist()
    Predict_Label = res[2].tolist()
    Value_test = Value
    Predict_Label_test = Predict_Label
    Value_test = np.array(Value_test)
    Predict_Label_test = np.array(Predict_Label_test)
    pcc_test = pearsonr(Value_test, Predict_Label_test)[0]
    n = len(Value_test)
    kde = gaussian_kde([Value_test, Predict_Label_test])
    density = kde([Value_test, Predict_Label_test])
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        Value_test, Predict_Label_test, 
        c=density, 
        cmap='jet', 
        s=10,       
        alpha=0.8   
    )

    divider = make_axes_locatable(ax) 
    cax = divider.append_axes("right", size="5%", pad=0.1)  

    vmin, vmax = density.min(), density.max()
    cbar = fig.colorbar(sc, cax=cax, ticks=np.linspace(vmin, vmax, 6))  
    cbar.set_label('Density', fontsize=12)
    cbar.ax.set_yticklabels([f'{v:.2f}' for v in np.linspace(vmin, vmax, 6)]) 
    ax.text(
        Value_test.min(), Predict_Label_test.max(), 
        f'PCC = {pcc_test:.2f}\nN = {n}', 
        va='top', ha='left', 
        fontsize=12, fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.8, pad=5)
    )
    ax.text(
        Value_test.max()*0.6, Predict_Label_test.max() * 0.9, 
        'pH dataset', 
        va='top', ha='left', 
        fontsize=12, fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.8, pad=5)
    )
    # ax.set_xlim(-4, 4)  # 限制x轴范围为0到120
    # ax.set_ylim(-4, 4)   # 限制y轴范围为0到80

    plt.xlabel(r'$\log_{10}$[experimental $k_{\text{cat}}$ value (s$^{-1}$)]')
    plt.ylabel(r'$\log_{10}$[predicted $k_{\text{cat}}$ value (s$^{-1}$)]')
    plt.tight_layout()
    plt.savefig('./UniKP_mindspore/metrics/kcat_correlation_scatter_ph_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def get_r2_score_3():
    # r2_scores_list = []
    # for i in range(5):
    #     res = np.array(pd.read_excel(f'./PreKcat_new/{i+1}_all_samples_metrics.xlsx', sheet_name='Sheet1')).T
    #     Training_test = res[9].tolist()
    #     Value = res[7].tolist()
    #     Predict_Label = res[8].tolist()
    #     Value_test = [Value[i] for i in range(len(Training_test)) if Training_test[i] == 1]
    #     Predict_Label_test = [Predict_Label[i] for i in range(len(Training_test)) if Training_test[i] == 1]
    #     Value_test = np.array(Value_test)
    #     Predict_Label_test = np.array(Predict_Label_test)
    #     r2_test = r2_score(Value_test, Predict_Label_test)
    #     r2_scores_list.append(r2_test)
    # average_r2 = np.mean(r2_scores_list)
    # ax = sns.boxplot(data=[r2_scores_list])
    # ax = sns.stripplot(data=[r2_scores_list], color='black', size=6, jitter=True, edgecolor="auto")
    # res = np.array(pd.read_excel(f'UniKP_mindspore/PreKcat_new/3_all_samples_metrics.xlsx', sheet_name='Sheet1')).T
    # Training_test = res[9].tolist()
    # Value = res[7].tolist()
    # Predict_Label = res[8].tolist()
    # Value_test = [Value[i] for i in range(len(Training_test)) if Training_test[i] == 1]
    # Predict_Label_test = [Predict_Label[i] for i in range(len(Training_test)) if Training_test[i] == 1]
    # Value_test = np.array(Value_test)
    # Predict_Label_test = np.array(Predict_Label_test)
    # r2_test = r2_score(Value_test, Predict_Label_test)
    # print('u' , r2_test)

    res_d = np.array(pd.read_excel(f'UniKP_mindspore/PreKcat_new/degree_Kcat_5_cv.xlsx', sheet_name='Sheet1')).T
    Value_d = res_d[1].tolist()
    Predict_Label_d = res_d[2].tolist()
    Value_test_d = np.array(Value_d)
    Predict_Label_test_d = np.array(Predict_Label_d)
    r2_test_d = r2_score(Value_test_d, Predict_Label_test_d)
    # r2_test_d = pearsonr(Value_test_d, Predict_Label_test_d)[0]
    print('d' , r2_test_d)

    res_p = np.array(pd.read_excel(f'UniKP_mindspore/PreKcat_new/ph_Kcat_5_cv.xlsx', sheet_name='Sheet1')).T
    Value_p = res_p[1].tolist()
    Predict_Label_p = res_p[2].tolist()
    Value_test_p = np.array(Value_p)
    Predict_Label_test_p = np.array(Predict_Label_p)
    r2_test_p = r2_score(Value_test_p, Predict_Label_test_p)
    # r2_test_p = pearsonr(Value_test_p, Predict_Label_test_p)[0]
    print('p' , r2_test_p)

    res_ds = np.array(pd.read_excel(f'UniKP_mindspore/PreKcat_new/s2_degree_Kcat_2.xlsx', sheet_name='Sheet1')).T
    Value_ds = res_ds[1].tolist()
    Predict_Label_ds = res_ds[7].tolist()
    Training_Validation_Test = res_ds[8].tolist()
    Value_test_ds = [Value_ds[i] for i in range(len(Training_Validation_Test)) if Training_Validation_Test[i] ==2]
    Predict_Label_test_ds = [Predict_Label_ds[i] for i in range(len(Training_Validation_Test)) if Training_Validation_Test[i] ==2]
    Value_test_ds = np.array(Value_test_ds)
    Predict_Label_test_ds = np.array(Predict_Label_test_ds)
    r2_test_ds = r2_score(Value_test_ds, Predict_Label_test_ds)
    # r2_test_ds = pearsonr(Value_test_ds, Predict_Label_test_ds)[0]
    print('ds' , r2_test_ds)

    res_ps = np.array(pd.read_excel(f'UniKP_mindspore/PreKcat_new/s2_ph_Kcat_2.xlsx', sheet_name='Sheet1')).T
    Value_ps = res_ps[1].tolist()
    Predict_Label_ps = res_ps[7].tolist()
    Training_Validation_Test = res_ps[8].tolist()
    Value_test_ps = [Value_ps[i] for i in range(len(Training_Validation_Test)) if Training_Validation_Test[i] ==2]
    Predict_Label_test_ps = [Predict_Label_ps[i] for i in range(len(Training_Validation_Test)) if Training_Validation_Test[i] ==2]
    Value_test_ps = np.array(Value_test_ps)
    Predict_Label_test_ps = np.array(Predict_Label_test_ps)
    r2_test_ps = r2_score(Value_test_ps, Predict_Label_test_ps)
    # r2_test_ps = pearsonr(Value_test_ps, Predict_Label_test_ps)[0]
    print('ps' , r2_test_ps)

    intervals = ['pH_EF_UniKP', 'pH_Revised_UniKP', 'temperature_EF_UniKP', 'temperature_Revised_UniKP']
    x = np.arange(len(intervals))  
    width = 0.35                   
    fig, ax = plt.subplots(figsize=(8, 5))

    rects_unikp = ax.bar(
        [x[0],x[2]], [r2_test_ps, r2_test_ds], 
        width, 
        color='#87CEEB'  
    )
    rects_unikp_e = ax.bar(
        [x[1],x[3]], [ r2_test_p, r2_test_d], 
        width, 
        color="#22B8F3"  
    )

    ax.set_xticks(x)
    ax.set_xticklabels(intervals)  
    ax.set_ylabel('R² on test set', fontsize=12)        
    ax.legend()                  
    def add_labels(rects, rects_e):
        for i in range(len(rects)):
            height = rects[i].get_height()
            ax.annotate(
                f'',
                xy=(rects[i].get_x() + rects[i].get_width()/2, height),
                xytext=(0, 3),  
                textcoords="offset points",
                ha='center', va='bottom'
            )
            height2 = rects_e[i].get_height()
            ax.annotate(
                f'',
                xy=(rects_e[i].get_x() + rects_e[i].get_width()/2, height),
                xytext=(0, 3),  
                textcoords="offset points",
                ha='center', va='bottom'
            )

    add_labels(rects_unikp , rects_unikp_e)
    plt.tight_layout()  
    plt.savefig('./UniKP_mindspore/metrics/er_r2.png', dpi=300, bbox_inches='tight')  
    plt.show()

    # plt.ylabel('R² on test set', fontsize=12)
    # plt.xticks([0], ['UniKP'])
    # plt.axhline(y=average_r2, color='r', linestyle='--', label=f'平均 R²: {average_r2:.4f}')
    # plt.legend()
    # plt.ylim(min(r2_scores_list) - 0.05, max(r2_scores_list) + 0.05)
    # plt.tight_layout()
    # plt.savefig("./metrics/r2_scores_boxplot.png", dpi=300, bbox_inches='tight') 
    # plt.show()

def get_rmse_numerical_intervals_re():
    Value_2, Predict_Label_2  = [],[]
    Value_1, Predict_Label_1  = [],[]
    Value_0, Predict_Label_0  = [],[]
    Value_neg, Predict_Label_neg  = [],[]
    intervals = ['CBW', 'CSW', 'DMW']

    res = np.array(pd.read_excel('UniKP_mindspore/PreKcat_new/3_0.9_Re_weighting_Kcat_5_cv.xlsx', sheet_name='Sheet1')).T
    Value = res[1].tolist()
    Predict_Label = res[2].tolist()
    Value_2 = [Value[i] for i in range(len(Value)) if Value[i] >= 4]
    Predict_Label_2 = [Predict_Label[i] for i in range(len(Value)) if Value[i] >= 4]
    print(Value_2)
    print(Predict_Label_2)
    res = np.array(pd.read_excel('UniKP_mindspore/PreKcat_new/Root_Cost_sensitive_Re_weighting_Kcat_5_cv.xlsx', sheet_name='Sheet1')).T
    Value = res[1].tolist()
    Predict_Label = res[2].tolist()
    Value_1 = [Value[i] for i in range(len(Value)) if Value[i] >= 4]
    Predict_Label_1 = [Predict_Label[i] for i in range(len(Value)) if Value[i] >= 4]
    res = np.array(pd.read_excel('UniKP_mindspore/PreKcat_new/DMW_No_Normalize_2_LDS_Kcat_5_cv.xlsx', sheet_name='Sheet1')).T
    Value = res[1].tolist()
    Predict_Label = res[2].tolist()
    Value_0 = [Value[i] for i in range(len(Value)) if Value[i] >= 4]
    Predict_Label_0 = [Predict_Label[i] for i in range(len(Value)) if Value[i] >= 4]
    rmse_2 = mean_squared_error(Value_2, Predict_Label_2)
    rmse_1 = mean_squared_error(Value_1, Predict_Label_1)
    rmse_0 = mean_squared_error(Value_0, Predict_Label_0)
    unikp_rmse = [ rmse_2, rmse_1, rmse_0]
    x = np.arange(len(intervals))  
    width = 0.35                   
    fig, ax = plt.subplots(figsize=(8, 5))

    rects_unikp = ax.bar(
        x, unikp_rmse, 
        width, 
        label='', 
        color="#025475"  
    )
    ax.set_xticks(x)
    ax.set_xticklabels(intervals)  
    ax.set_ylabel('RMSE', fontsize=12)        
    ax.set_xlabel(r'$\log_{10}$[experimental $k_{\text{cat}}$ value (s$^{-1}$)]', fontsize=12)
    ax.legend()                  
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f'{height:.4f}',
                xy=(rect.get_x() + rect.get_width()/2, height),
                xytext=(0, 3),  
                textcoords="offset points",
                ha='center', va='bottom'
            )

    add_labels(rects_unikp)
    plt.tight_layout()  
    plt.savefig('UniKP_mindspore/metrics/rmse_numerical_intervals_re.png', dpi=300, bbox_inches='tight')  
    plt.show()

if __name__ == '__main__':

    # get_r2_score()
    # get_rmse_score()
    # get_pcc_scatter_plot()
    # get_rmse_numerical_intervals()
    # get_wildtype_scatter_plot()
    # get_mutant_scatter_plot()
    # get_pcc_wild_mutant()
    get_rmse_numerical_intervals_re()