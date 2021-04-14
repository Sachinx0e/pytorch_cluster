def plot_roc_curve(fig,ax,roc_curve_data,label):
    
    fps = roc_curve_data[0]
    tpr = roc_curve_data[1]

    ax.plot(fps, tpr, linestyle='--', label=label)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    fig.legend(loc='upper right',
               prop={'size': 6}
            )

def plot_roc_curve_baseline(fig,ax):
    ax.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    fig.legend(loc='upper right',
               prop={'size': 6}
    )

def plot_pr_curve(fig,ax,pr_curve_data,label):
    
    precision = pr_curve_data[0]
    recall = pr_curve_data[1]

    ax.plot(recall, precision, linestyle='--', label=label)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    fig.legend(loc='upper right',
               prop={'size': 6}
            )
    #fig.tight_layout()

def plot_pr_curve_baseline(fig,ax,pr_baseline):

    no_skill = pr_baseline
    ax.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    fig.legend(loc='upper right',
               prop={'size': 6}
            )