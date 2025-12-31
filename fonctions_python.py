import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf

import pickle # Ne marche pas avec Heroku
import joblib # Marche avec Heroku

import lightgbm as lgb
from lightgbm import LGBMClassifier

import gc  # Pour la gestion de la mémoire


''' ------------------- '''
''' Fonctions générales '''
''' ------------------- '''
def save_pickle(obj, filename, filepath):
    with open(f'{filepath}{filename}', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename, filepath):
    with open(f'{filepath}{filename}', 'rb') as handle:
        return pickle.load(handle)
    

def save_joblib(obj, filename, filepath):
    joblib.dump(obj, f'{filepath}{filename}')


def load_joblib(filename, filepath):
    return joblib.load(f'{filepath}{filename}') 


def data_duplicated(df):
    '''Retourne le nombres de lignes identiques.'''
    return df.duplicated(keep=False).sum()


def missing_cells(df):
    '''Calcule le nombre de cellules manquantes sur le data set total'''
    return df.isna().sum().sum()


def missing_cells_perc(df):
    '''Calcule le pourcentage de cellules manquantes sur le data set total'''
    return df.isna().sum().sum()/(df.size)


def missing_general(df):
    '''Donne un aperçu général du nombre de données manquantes dans le data frame'''
    print('Nombre total de cellules manquantes :',missing_cells(df))
    print('Nombre de cellules manquantes en % : {:.2%}'.format(missing_cells_perc(df)))


def valeurs_manquantes(df):
    '''Prend un data frame en entrée et créer en sortie un dataframe contenant le nombre de valeurs manquantes
    et leur pourcentage pour chaque variables. '''
    tab_missing = pd.DataFrame(columns = ['Variable', 'Missing values', 'Missing (%)'])
    tab_missing['Variable'] = df.columns
    missing_val = list()
    missing_perc = list()
    
    for var in df.columns:
        nb_miss = missing_cells(df[var])
        missing_val.append(nb_miss)
        perc_miss = missing_cells_perc(df[var])
        missing_perc.append(perc_miss)
        
    tab_missing['Missing values'] = list(missing_val)
    tab_missing['Missing (%)'] = list(missing_perc)
    return tab_missing


def bar_missing(df):
    '''Affiche le barplot présentant le nombre de données présentes par variable.'''
    msno.bar(df)
    plt.title('Nombre de données présentes par variable', size=15)
    plt.show()


def barplot_missing(df):
    '''Affiche le barplot présentant le pourcentage de données manquantes par variable.'''
    proportion_nan = df.isna().sum().divide(df.shape[0]/100).sort_values(ascending=False)
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 30))
    ax = sns.barplot(y = proportion_nan.index, x=proportion_nan.values)
    plt.title('Pourcentage de données manquantes par variable', size=15)
    plt.show()


def drop_columns_empty(df, lim):
    '''Prend en entrée un data frame et un seuil de remplissage de données.
    Supprime chaque feature ayant un pourcentage de données manquantes supérieur à celui renseigné. 
    Donne en sortie le data frame filtré avec les colonnes à garder.'''
    
    tab = valeurs_manquantes(df)
    columns_keep = list()
    for row in tab.iterrows():
        if float(row[1]['Missing (%)']) > float(lim):
            print('Suppression de la feature {} avec un % de valeurs manquantes à {}'.format(row[1]['Variable'], round(float(row[1]['Missing (%)']), 2)))
        else :
            columns_keep.append(row[1]['Variable'])
    
    return df[columns_keep]


''' -------------------- '''
''' Fonctions graphiques '''
''' -------------------- '''
def distribution(df, colonnes, n_cols, fig=(20, 20)):
    ''' Affiche les histogrammes pour chaque variable renseignée.'''
    n_rows = int(len(colonnes)/n_cols) + 1
    fig = plt.figure(figsize=fig)
    for i, col in enumerate(colonnes, 1):
        ax = fig.add_subplot(n_rows, n_cols, i)
        sns.histplot(data=df, x=col, bins=30, kde=True, ax=ax)

    plt.tight_layout(pad = 2)
    plt.show()


def bar_plot(df, colonnes, n_cols, fig=(20,20)):
    ''' Affiche les bar plots pour chaque variable renseignée.'''
    fig = plt.figure(figsize=fig)
    n_rows = int(np.ceil(len(colonnes)/n_cols))
    for i, col in enumerate(colonnes,1):
        ax = fig.add_subplot(n_rows,n_cols,i)
        count = df[col].value_counts()
        count.plot(kind="bar", ax=ax, fontsize=20, rot=90)
        ax.set_title(col, fontsize = 20)
    plt.tight_layout(pad = 2)
    plt.show()


def bar_plot_stacked(df, colonnes, n_cols, fig=(20,20)):
    ''' Affiche les bar plots pour chaque variable renseignée décomposés en fonction de var2.'''
    fig = plt.figure(figsize=fig)
    n_rows = int(np.ceil(len(colonnes) / n_cols))
    for i, col in enumerate(colonnes, 1):
        ax = fig.add_subplot(n_rows, n_cols, i)
        count = pd.DataFrame(df.groupby(col)['TARGET'].value_counts()).reset_index()
        count = count.pivot_table(index=col, columns = 'TARGET', values = 'count')
        #count = pd.crosstab(index=count[col], columns =count[var2], values = 'count')
        count.plot(kind="bar", stacked=True, ax=ax, fontsize=20, rot=70)
        ax.set_title(col, fontsize = 20)
        ax.legend(['rembourse', 'défaut'], fontsize = 20)
    plt.tight_layout(pad = 2)
    plt.show()


def pie_plot(df,colonnes):
    '''Affiche un pie plot présentant la répartition de la variable renseignée.'''
    for col in colonnes :
        labels = list(df[col].value_counts().sort_index().index.astype(str))
        count = df[col].value_counts().sort_index()
        
        plt.figure(figsize=(5, 5))
        plt.pie(count,autopct='%1.2f%%')
        plt.title('Répartition de {}'.format(col), size = 20)
        plt.legend(labels)
        plt.show()


def distribution_densite(df, colonnes, n_cols, fig=(20,20)):
    ''' Affiche les densités  pour chaque feature renseignée.'''
    n_rows = int(len(colonnes)/n_cols)+1
    fig = plt.figure(figsize=fig)
    for i, col in enumerate(colonnes,1):
        ax = fig.add_subplot(n_rows,n_cols,i)
        sns.kdeplot(df.loc[df['TARGET'] == 0, col], label = 'Client qui rembourse')
        sns.kdeplot(df.loc[df['TARGET'] == 1, col], label = 'Client en défaut')
        ax.set_xlabel(col, fontsize=20)
        ax.set_ylabel('Densité', fontsize=20)
        ax.legend(fontsize=20)
        ax.set_title('Distribution de ' + col, fontsize=20)
        
    plt.tight_layout(pad = 2)
    plt.show()
    
    
def heat_map(df_corr):
    '''Affiche la heatmap '''
    plt.figure(figsize=(30,30))
    sns.heatmap(df_corr, annot=True, linewidth=.5)
    plt.title("Heatmap")


def fgraph_prediction(y_test, y_pred_test, figsize=(20, 10)):
    # figure
    plt.figure(figsize=figsize)

    # residuals
    y_resid = (y_test - y_pred_test)
    
    # plotting the histplot
    plt.subplot(221)
    plt.title("Histplot")
    plt.grid(True)
    sns.histplot(y_resid)
    
    # plotting the residual plot
    plt.subplot(222)
    plt.title("Residual Plot")
    plt.grid(True)
    sns.scatterplot(data=None, x=y_test, y=y_resid)
    sns.lineplot(data=None, x=[min(y_test), max(y_test)], y=[0, 0], linestyle = '--', color = 'r')
    
    # plotting the quantile plot
    ax = plt.subplot(223)
    plt.title("Quantile Plot")
    plt.grid(True)
    sm.qqplot(y_resid, line = 'r', ax = ax)
    
    # plotting the autocorrelation plot
    ax = plt.subplot(224)
    plt.title("Autocorrelation Plot")
    plt.grid(True)
    plot_acf(y_resid, ax = ax)

    plt.show()


def fgraph_feature_importance(model, X_test, y_test, featureY, figsize=(20,15)):
    # Feature importance :
    df_feature_importance = pd.DataFrame(columns=["Feature Name", "Feature Importance"])
    df_feature_importance["Feature Name"] = pd.Series(model.feature_names_in_)
    df_feature_importance["Feature Importance"] = pd.Series(model.feature_importances_)
    df_feature_importance = df_feature_importance.loc[(df_feature_importance["Feature Importance"] > 0.001), :].sort_values(by="Feature Importance", ascending=True)

    # Permutation importance :
    perm = permutation_importance(
        model, 
        X_test,
        y_test,
        scoring= "r2",
        n_repeats=5, 
        random_state=0,
    )

    df_permutation_importance = pd.DataFrame(columns=["Feature Name", "Feature Importance", "Std Importance"])
    df_permutation_importance["Feature Name"] = pd.Series(model.feature_names_in_)
    df_permutation_importance["Feature Importance"] = pd.Series(perm["importances_mean"])
    df_permutation_importance = df_permutation_importance.loc[(df_permutation_importance["Feature Importance"] > 0.001), :].sort_values(by="Feature Importance", ascending=True)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    plt.suptitle(f"Feature Importance de {featureY}", fontsize=22)
    df_feature_importance.plot.barh(x="Feature Name", y="Feature Importance", color="steelblue", ax=axes[0])
    axes[0].set_xlabel("Feature importance", fontsize=14)
    axes[0].set_ylabel("", fontsize=0)
    df_permutation_importance.plot.barh(x="Feature Name", y="Feature Importance", color="steelblue", ax=axes[1])
    axes[1].set_xlabel("Permutation importance", fontsize=14)
    axes[1].set_ylabel("", fontsize=0)
    fig.tight_layout()
    plt.show()


def missing_data(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum()/data.isnull().count()*100)\
        .sort_values(ascending=False)
    return pd.concat([total, percent],
                     axis=1,
                     keys=['Total', 'Percent'])


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category=False):
    original_columns = list(df.columns)
    categorical_columns = [
        col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns,
                        dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm(df, num_folds, stratified = False, debug= False):
    #df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    #test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}".format(train_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    #sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            #nthread=4,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            verbose=-1, )

        clf.fit(train_x, train_y,
                eval_set=[(train_x, train_y), (valid_x, valid_y)], 
                eval_metric= 'auc',
                callbacks=[lgb.callback.early_stopping(stopping_rounds=200),
                           lgb.callback.log_evaluation(200)])

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        #sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.4f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])), sep='')
        print("\n")
     
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.4f' % roc_auc_score(train_df['TARGET'], oof_preds))

    display_importances(feature_importance_df)
    
    return feature_importance_df


# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False).index  #[:40]
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()


# Replace string
def replace_string(phrase_complete, phrase_a_retirer):
    return phrase_complete.replace(phrase_a_retirer, "")
