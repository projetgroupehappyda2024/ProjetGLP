#Projet Data Analyse sur le Bien-Etre - réalisé par Lodaodavé LEMA, Gaëlle MARINESQUE, Hélène KHALIDI et Patrick BOUKÉ 
#DataScientest CDA Juin 2024. 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_graphviz
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from sklearn import tree
import graphviz
import io
import plotly.graph_objects as go

@st.cache_data
def load_data():
    try:
        data = pd.read_csv("happycommun1.csv")
        data_avant = pd.read_csv("happy_avant_fusion.csv")
        return data, data_avant
    except FileNotFoundError:
        st.error("Le fichier 'happycommun1.csv' est introuvable")
        return None 

data, data_avant = load_data()


st.sidebar.image("https://myfiteo.app/src_img_happy/img0.png", use_column_width=True)
st.sidebar.markdown("<h1 style='font-weight:normal; color:#000000;'><b>Le Bonheur dans le monde</b></h1>", unsafe_allow_html=True)
st.sidebar.markdown('<hr>', unsafe_allow_html=True)
menu = st.sidebar.radio("",["Introduction", "Hypothèses", "Sources", "Visualisation", "Modélisation", "Analyse Machine Learning", "Conclusion"])
st.sidebar.markdown('<hr>', unsafe_allow_html=True)
st.sidebar.markdown("<h4>🌍 Projet réalisé par:</h4>  <p>Lodavé LEMA, Gaëlle MARINESQUE, <br>Hélène KHALIDI et Patrick BOUKÉ</p><h4>Promotion DA - Juin 2024</h4>", unsafe_allow_html=True)   

if data is not None:  

    

    if menu == "Introduction":
        st.image("https://myfiteo.app/src_img_happy/img3.png", use_column_width=True)
        
        
        st.markdown(
        """
        <h1 style="
            font-size: 2em; 
            margin: 0; 
            padding: 0;">
            Analyse du Bonheur et du Bien-être
        </h1><br>
        <p align="justify"><i>La recherche du bonheur est certainement la plus grande quête de l’humanité. Nous cherchons tous à être heureux. L’analyse des indices de bonheur nous offre un aperçu précieux de ce qui rend les individus et les sociétés réellement épanouis. Grâce aux données, nous ne nous contentons pas de mesurer un sentiment subjectif : nous identifions les tendances, comprenons les disparités et révélons les leviers qui favorisent un bien-être durable.</i></p>

        <br>
        <br>
        """,
        unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Problématique", "Recherches principales"])

        

        with tab1:
            st.markdown("### Problématique", unsafe_allow_html=True)
            st.markdown("""<p align="justify">Le bonheur des populations est un indicateur complexe et multidimensionnel qui va bien au-delà des simples mesures économiques. Il intègre des dimensions sociales, économiques et politiques qui interagissent de manière complexe pour influencer le bien-être global des individus. Cette étude cherche à approfondir la compréhension de ces facteurs et à déterminer comment ils contribuent aux variations de bonheur entre les pays sur la période de 2005 à 2023.</p>""",unsafe_allow_html=True)

        with tab2:
            st.markdown("### Questions de recherches principales", unsafe_allow_html=True)
            st.markdown("1. Quels sont les principaux facteurs sociaux, économiques et politiques qui influencent le bonheur mondial ?",unsafe_allow_html=True)
            st.markdown("2. Comment ces facteurs ont-ils évolué à travers le temps en Europe ?",unsafe_allow_html=True)
            st.markdown("3. L'argent fait-il le bonheur ?",unsafe_allow_html=True)

    
    elif menu == "Hypothèses":
        st.image("https://myfiteo.app/src_img_happy/img2.png", use_column_width=True)
    
        st.markdown("<h1>Hypothèses</h1>", unsafe_allow_html=True)
    
        
        tab1, tab2 = st.tabs(["Hypothèses générales", "Hypothèses spécifiques"])
    
        with tab1:
            st.markdown("### Hypothèses générales (au niveau mondial)", unsafe_allow_html=True)
            st.markdown("1. Les variables économiques telles que le PIB par habitant ont un impact significatif sur le bonheur des populations.", unsafe_allow_html=True)
            st.markdown("2. Les variables sociales, telles que l'accompagnement social et l'espérance de vie en bonne santé, jouent un rôle crucial dans la satisfaction de vie des individus.", unsafe_allow_html=True)
            st.markdown("3. Les dimensions de la liberté (économique, personnelle et choix de vie) influencent positivement le bonheur mondial.", unsafe_allow_html=True)
            st.markdown("4. La perception de la corruption et les affects positifs et négatifs sont également des déterminants importants du bonheur.", unsafe_allow_html=True)
            st.markdown("5. Les tendances annuelles du bonheur montrent des variations significatives entre les continents, influencées par des facteurs régionaux spécifiques.", unsafe_allow_html=True)
    
        with tab2:
            st.markdown("### Hypothèses spécifiques (concentrées sur l'Europe)", unsafe_allow_html=True)
            st.markdown("1. En Europe, les facteurs sociaux tels que le soutien social et les politiques de protection sociale ont un impact plus prononcé sur le bonheur par rapport aux autres régions du monde.", unsafe_allow_html=True)
            st.markdown("2. En Europe, les niveaux élevés de liberté économique et personnelle sont fortement corrélés avec des niveaux élevés de satisfaction de vie.", unsafe_allow_html=True)
            st.markdown("3. Les interactions entre les variables économiques (PIB, liberté économique), sociales (soutien social, santé) et politiques (perception de la corruption) expliquent en grande partie les variations de bonheur entre les pays européens.", unsafe_allow_html=True)



    elif menu == "Sources":
        st.image("https://myfiteo.app/src_img_happy/img1.png", use_column_width=True)
        
        st.markdown(
        """
        <h1 style="
            font-size: 2em; 
            margin: 0; 
            padding: 0;">
            Sources
        </h1>
        """,
        unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["Données initiales", "Aperçu des données", "Données complémentaires", "Aperçu des données fusionnées"])

        with tab1:
            st.markdown("### Données initiales", unsafe_allow_html=True)
            st.markdown("""<p align="justify">Dans le cadre de l’analyse exploratoire nous avons eu accès à une base de données (DataForTable2.1.xls) disponible librement sur <a href="https://worldhappiness.report/data/">le site web World Happiness</a> des Nations Unies. Elle regroupe différentes enquêtes réalisées à travers le monde :</p>""", unsafe_allow_html=True)
            st.markdown("  - Note de bonheur (Life Ladder)", unsafe_allow_html=True)
            st.markdown("  - PIB par habitant (Log GDP per capita)", unsafe_allow_html=True)
            st.markdown("  - Support social (Social support)", unsafe_allow_html=True)
            st.markdown("  - Espérance de vie à la naissance (Healthy life expectancy at birth)", unsafe_allow_html=True)
            st.markdown("  - Liberté de faire des choix de vie (Freedom to make life choices)", unsafe_allow_html=True)
            st.markdown("  - Générosité (Generosity)", unsafe_allow_html=True)
            st.markdown("  - Perception de la corruption (Perceptions of corruption)", unsafe_allow_html=True)
            st.markdown("  - Affects positifs et négatifs (Positive affect / Negative affect)", unsafe_allow_html=True)

        with tab2:
            st.markdown("### Aperçu des données", unsafe_allow_html=True)
            st.dataframe(data_avant.head(5))

        with tab3:
            st.markdown("### Données complémentaires", unsafe_allow_html=True)
            st.markdown("""<p align="justify">Pour enrichir la base de données, une nouvelle enquête (2023-Human-Freedom-Index-Data.xls) a été intégrée depuis <a href="https://www.cato.org/human-freedom-index/2023">le site web Cato Institute</a>. Elle apporte une notion différente dans cette analyse. Celle-ci note la liberté humaine au travers de 2 grands principes :</p>""", unsafe_allow_html=True)
            st.markdown("  - Liberté personnelle (PERSONAL FREEDOM) avec les aspects légaux, sécurité, mouvement des individus, religion, politique et presse.", unsafe_allow_html=True)
            st.markdown("  - Liberté économique (ECONOMIC FREEDOM) avec les aspects d’ordre politiques, égalité hommes/femmes, monétaire, échanges internationaux et règlementation du travail.", unsafe_allow_html=True)
            st.markdown("Il paraît pertinent d’inclure les notes de liberté personnelle et de liberté économique car elles découlent de données issues d’études concrètes et pertinentes par rapport à l’analyse du bonheur.", unsafe_allow_html=True)

        with tab4:
            st.markdown("### Aperçu des données fusionnées", unsafe_allow_html=True)
            st.dataframe(data.head(5))

    
    
    elif menu == "Visualisation":
        
    
        #st.image("https://myfiteo.app/src_img_happy/img4.png", use_column_width=True)
        st.markdown("<h1>Visualisation des Données</h1>", unsafe_allow_html=True)
        st.markdown("<h3>Au niveau Mondial</h3>", unsafe_allow_html=True)
        st.image("https://myfiteo.app/src_img_happy/img9.png", use_column_width=True)
        
        
    
        # Section Carte Interactive
        with st.expander("Carte interactive de l'indice de bonheur par pays", expanded=False):
            if "year" in data.columns and "Country name" in data.columns and "Life Ladder" in data.columns:
                st.markdown("### Evolution de l'indice de bonheur par pays")
                happy_by_year = data.sort_values(by="year")
                
                fig = px.choropleth(
                    happy_by_year,
                    locations="Country name",
                    locationmode="country names",
                    color="Life Ladder",
                    color_continuous_scale="rainbow",
                    animation_frame="year",
                    labels={"year": "Année"},
                    range_color=[0, 10]
                )
                
                fig.update_layout(
                    geo=dict(showframe=False, showcoastlines=True, projection_type='miller'),
                    coloraxis_colorbar=dict(title="Life Ladder"),
                    width=800,
                    height=800
                )
                
                st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})
            else:
                st.error("Les colonnes nécessaires pour créer la carte interactive sont manquantes.")
    
        # Section Heatmap des Corrélations
        with st.expander("Heatmap des Corrélations"):
            st.markdown("<h4>Corrélation entre le bonheur (life ladder) et les autres indicateurs</h4>", unsafe_allow_html=True)
            
            if data is not None:
                num = data.iloc[:, 4:15].corr()
                fig, ax = plt.subplots(figsize=(10, 10))
                sns.heatmap(num, annot=True, ax=ax, cmap='rainbow')
                st.pyplot(fig)
    
        # Section Boxplot
        with st.expander("Vérification des valeurs extrêmes avec les Boxplots"):
            if data is not None:
                happyfiltre = data.dropna(subset=['Life Ladder', 'Economic Freedom', 'Log GDP per capita', 'Social support', 'Healthy life expectancy at birth'])
                happyfiltre = happyfiltre[~happyfiltre['year'].isin([2005, 2022, 2023])].round(3)
                happyfiltre['Social support'] = happyfiltre['Social support'] * 10
                happyfiltre["Healthy life expectancy at birth"] = happyfiltre["Healthy life expectancy at birth"]/10
    
                st.markdown("<h4>Vérification des valeurs extrêmes de ces variables</h4>", unsafe_allow_html=True)
                st.markdown("""<p align="justify"><b>Social Support : </b> Ces valeurs extrêmes pourraient correspondre à des pays ou des régions où le soutien social est significativement moins développé ou peu accessible.<br><br><b>Healthy Life Expectancy at Birth :</b> Ces valeurs extrêmes faibles peuvent être associées à des pays en développement ou à des régions confrontées à des défis sanitaires majeurs. Cette disparité importante est probablement influencée par des facteurs comme l'accès aux soins, la qualité de vie et les infrastructures sanitaires.<br></p>""", unsafe_allow_html=True)
                col = ['Life Ladder', 'Economic Freedom', 'Log GDP per capita', 'Social support', 'Healthy life expectancy at birth']
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(data=happyfiltre[col], ax=ax, color="#4498e8")
                #ax.set_title("Boxplots des Variables économiques et sociales")
                st.pyplot(fig)
    
        # Section Moyenne des Variables par Continent
        with st.expander("Moyenne des variables par continent"):
            if data is not None:
                moyenne_par_continent = happyfiltre.groupby('Continent')[['Life Ladder', 'Economic Freedom', 'Log GDP per capita', 'Social support', 'Healthy life expectancy at birth']].mean().reset_index()
                data_melted = moyenne_par_continent.melt(id_vars='Continent', var_name='Variable', value_name='Moyenne')
    
                st.markdown("### Moyenne des variables par continent", unsafe_allow_html=True)
                st.markdown("""<p align="justify">Les continents Océanie, Amérique et Europe, bénéficient des meilleurs indices  de bonheur (Life Ladder), de liberté économique (Economic Freedom), de PIB par habitant (Log GDP per capita) et de soutien social (Social support). On peut remarquer que sur certaines variables l’Europe est talonnée par l’Amérique du Sud.</p>""", unsafe_allow_html=True)
                
                fig, ax = plt.subplots(figsize=(14, 8))
                sns.barplot(data=data_melted, x='Continent', y='Moyenne', hue='Variable', palette='rainbow', ax=ax)
                ax.grid(True, which='both', linestyle='--', linewidth= 0.7)
                #ax.set_title("Moyenne des variables par continent", fontsize=16)
                plt.tight_layout()
                st.pyplot(fig)
    
        # Section Évolution par Continent
        with st.expander("Évolution de l'indice du bonheur par continent"):
            st.markdown("### Évolution du bonheur par continent")
            st.markdown("Ce graphique permet de comparer l’évolution de l’indice du bonheur au fil des années entre les continents")
            moy_continent_an = happyfiltre.groupby(['Continent', 'year'])[['Life Ladder', 'Economic Freedom', 'Log GDP per capita', 'Social support', 'Healthy life expectancy at birth']].mean().reset_index()
            fig = px.line(
                moy_continent_an,
                x="year",
                y="Life Ladder",
                color="Continent",
                markers=True,
                labels={"year": "Année", "Life Ladder": "Indice de bonheur", "Continent": "Continent"}
            )
            fig.update_layout(width=1000, height=600, template="plotly_white", xaxis=dict(tickmode="linear", dtick=1))
            
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': False})

        
        st.markdown("<h3>Au niveau Européen</h3>", unsafe_allow_html=True)
        st.image("https://myfiteo.app/src_img_happy/img12.png", use_column_width=True)

        
        # Section Évolution en Europe
        with st.expander("Évolution de l'indice de bonheur en Europe"):
            happy_europe = happyfiltre[happyfiltre['Continent'] == 'Europe'].sort_values(by='year')
            happy_europe = happy_europe.groupby(['Continent', 'Regional indicator', 'year'])[['Life Ladder', 'Log GDP per capita', 'Economic Freedom', 'Social support', 'Healthy life expectancy at birth']].mean(numeric_only=True).reset_index()
        
            
            st.markdown("### Évolution de l'indice de bonheur en Europe")
            st.markdown(
            """
            <style>
                .year {
                    color: #383838;
                    font-weight: bold;
                }
                .pin {
                    display: inline-block;
                    width: 10px;
                    height: 10px;
                    background-color: #ec5a53;
                    border-radius: 50%;
                    margin-right: 8px;
                    
                }
            </style>
            <div>
                <div><span class="year">Repères:</span></div>
                <div><span class="pin"></span><span class="year">2008 :</span> Pré-crises des subprimes, perception d’une prospérité continue, croissance économique soutenue, stabilité économique. D’autres facteurs ont pu influencer l’indice du bonheur, ex : JO de Pékin (un sentiment de fierté collective et une perception positive de la société).</div>
                <div><span class="pin"></span><span class="year">2009 :</span> Conséquence de la crise économique sur le bonheur.</div>
                <div><span class="pin"></span><span class="year">2010 :</span> Pic en Europe du Nord pouvant être expliqué par la mise en place de politiques économiques, sociales et écologiques renforçant le sentiment de bien-être général.</div>
                <div><span class="pin"></span><span class="year">2020 – 2022 :</span> COVID.</div>
            </div>
            """,
            unsafe_allow_html=True)
            custom_palette = ['#FF5733', '#33FF57', '#3357FF', '#FFC300','#DAF7A6']
            fig = px.line(
                happy_europe,
                x="year",
                y="Life Ladder",
                color="Regional indicator",
                markers=True,
                color_discrete_sequence=custom_palette,
                #title="Évolution de l'indice de bonheur par sous-régions européennes",
                labels={"year": "Évolution de l'indice de bonheur par sous-régions européennes", "Life Ladder": "Indice de bonheur", "Regional indicator": "Sous-régions"}
            )
            fig.update_layout(
                width=800,
                height=400,
                legend_title_text="Sous-régions",
                template="plotly_white",
                xaxis=dict(tickmode="linear", tick0=2005, dtick=1)
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': False})
    
        # Section Évolution des Variables en Europe
        with st.expander("Évolution des variables au fil du temps en Europe"):
            happy_EU_variable = happy_europe.groupby(['Continent', 'year'])[['Life Ladder', 'Economic Freedom', 'Log GDP per capita', 'Social support', 'Healthy life expectancy at birth']].mean(numeric_only=True).reset_index()
            
                
            data_long = happy_EU_variable.melt(
                id_vars="year", 
                value_vars=["Life Ladder", "Log GDP per capita", "Economic Freedom", "Social support", 'Healthy life expectancy at birth'], 
                var_name="Variable", 
                value_name="Value"
            )
        
            
            st.markdown("### Évolution des variables au fil du temps en Europe")
            st.markdown("Les variables sont globalement stables")
            custom_palette2 = ['#FF5733', '#33FF57', '#3357FF', '#FFC300','#DAF7A6']
            fig = px.line(
                data_long,
                x="year",
                y="Value",
                color="Variable",
                markers=True,
                color_discrete_sequence=custom_palette2,
                title="",
                labels={"Value": "Valeurs Moyennes", "year": "Évolution des variables au fil du temps en Europe", "Variable": "Variables"}
            )
            fig.update_layout(
                width=800, 
                height=400, 
                legend=dict(title="Variables"),
                xaxis=dict(tickmode="linear", tick0=2005, dtick=1),
                template="plotly_white",
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': False})
        
       

    

    elif menu == "Modélisation":
        st.image("https://myfiteo.app/src_img_happy/img8.png", use_column_width=True)
        st.markdown("<h1>Modélisation</h1>", unsafe_allow_html=True)
        st.markdown('<hr>', unsafe_allow_html=True)
    
        
        try:
            happyeurope = pd.read_csv('happycommun1.csv')
            happyeurope = happyeurope.loc[happyeurope['Continent']=='Europe'].drop('Continent', axis=1)
        except Exception as e:
            st.error(f"Erreur lors du chargement des données : {str(e)}")
            st.stop()
    
        try:
            # Préparation des données
            feats = happyeurope.drop('Life Ladder', axis=1)
            target = happyeurope['Life Ladder']
    
            # Split des données
            X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.25, random_state=42)
    
            # Traitement des données manquantes et preprocessing
            def preprocess_data(X_train, X_test):
                columns_to_impute = [
                    'Log GDP per capita', 'Social support', 'Healthy life expectancy at birth',
                    'Freedom to make life choices', 'Generosity', 'Perceptions of corruption',
                    'Positive affect', 'Negative affect', 'Economic Freedom', 'Personal Freedom'
                ]
                
                # Imputation par région et année
                for column in columns_to_impute:
                    median_values = X_train.groupby(['Regional indicator', 'year'])[column].median()
                    for (regions, year), median in median_values.items():
                        X_train.loc[(X_train['Regional indicator'] == regions) & 
                                   (X_train['year'] == year) & 
                                   (X_train[column].isna()), column] = median
                        X_test.loc[(X_test['Regional indicator'] == regions) & 
                                  (X_test['year'] == year) & 
                                  (X_test[column].isna()), column] = median
    
                # Imputation des valeurs restantes par région
                for column in columns_to_impute:
                    median_values = X_train.groupby(['Regional indicator'])[column].median()
                    for regions, median in median_values.items():
                        X_train.loc[(X_train['Regional indicator'] == regions) & 
                                   (X_train[column].isna()), column] = median
                        X_test.loc[(X_test['Regional indicator'] == regions) & 
                                  (X_test[column].isna()), column] = median
    
                # Scaling des données numériques
                numeric_cols = columns_to_impute
                scaler = MinMaxScaler(feature_range=(0, 10))
                X_train['year'] = X_train['year'].astype("int")
                X_test['year'] = X_test['year'].astype("int")
                X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
                X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
                # Encoding des variables catégorielles
                columns_to_encode = ['Regional indicator', 'Country name']
                encoder = OneHotEncoder(drop='first', sparse_output=False)
                
                encoded_colstrain = encoder.fit_transform(X_train[columns_to_encode])
                encoded_colstest = encoder.transform(X_test[columns_to_encode])
                
                encoded_col_names = encoder.get_feature_names_out(columns_to_encode)
                
                # Création des DataFrames finaux
                encoded_dftrain = pd.DataFrame(encoded_colstrain, columns=encoded_col_names, index=X_train.index)
                encoded_dftest = pd.DataFrame(encoded_colstest, columns=encoded_col_names, index=X_test.index)
                
                remaining_dftrain = X_train.drop(columns=columns_to_encode)
                remaining_dftest = X_test.drop(columns=columns_to_encode)
                
                return pd.concat([encoded_dftrain, remaining_dftrain], axis=1), pd.concat([encoded_dftest, remaining_dftest], axis=1)
    
            # Prétraitement des données
            Xtrain, Xtest = preprocess_data(X_train, X_test)


    
            # Interface utilisateur pour la modélisation
            st.write("### Paramètres de modélisation")
            
            col1, col2 = st.columns(2)
            with col1:
                model_choice = st.selectbox(
                    "Sélectionnez un modèle",
                    ["Decision Tree","Régression Linéaire","Random Forest", "XGBoost"], 
                    #format_func=lambda x: "Aucun" if x is None else x
                )
                desactivation_param = st.checkbox("Désactiver les paramètres")
            
            with col2:
                if model_choice in ["Decision Tree", "Random Forest", "XGBoost"] and not desactivation_param:
                    max_depth = st.slider("Profondeur maximale", 1, 10, 3)
                    min_samples_leaf = st.slider("Nombre minimum d'échantillons par feuille", 1, 50, 25)
            st.sidebar.markdown('<hr>', unsafe_allow_html=True)
            if model_choice and st.button("Lancer l'entraînement"):
                with st.spinner("Entraînement du modèle en cours..."):
                    def train_and_evaluate_model():
                        # Sélection du modèle
                        if model_choice == "Régression Linéaire":
                            model = LinearRegression()
                        elif model_choice == "Decision Tree" and not desactivation_param:
                            model = DecisionTreeRegressor(max_depth=max_depth, 
                                                       min_samples_leaf=min_samples_leaf, 
                                                       random_state=42)
                        elif model_choice == 'Decision Tree':
                            model = DecisionTreeRegressor(max_depth=3, random_state=42)
                        elif model_choice == 'Random Forest' and not desactivation_param:
                            model = RandomForestRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42, n_estimators=100)
                        elif model_choice == 'Random Forest':
                            model = RandomForestRegressor(max_depth=3, random_state=42, n_estimators=100)
                            
                        elif model_choice == 'XGBoost' and not desactivation_param:
                            model = XGBRegressor(max_depth=max_depth, min_child_weight=min_samples_leaf, random_state=42, n_estimators=100)

                        else: 
                            model = XGBRegressor(max_depth=3, random_state=42, n_estimators=100)

                        st.markdown('<hr>', unsafe_allow_html=True)
                            
                        
                        
                        # Entraînement et prédictions
                        model.fit(Xtrain, y_train)
                        y_pred_train = model.predict(Xtrain)
                        y_pred_test = model.predict(Xtest)
                        
                        return model, y_pred_train, y_pred_test
    
                    # Entraînement et évaluation du modèle
                    model, y_pred_train, y_pred_test = train_and_evaluate_model()



                  # explication
                    if model_choice == "Decision Tree":
                        st.info("""
                        💡 L'importance d'une variable dans un arbre de décision est calculée en fonction de la réduction 
                        de l'impureté (MSE dans ce cas) qu'elle apporte lors des divisions. Plus une variable permet de 
                        réduire l'erreur, plus elle est considérée comme importante.
                        """)
                    elif model_choice == "Random Forest":
                        st.info("""
                        💡 Dans un Random Forest, l'importance d'une variable est moyennée sur tous les arbres.
                        Cela donne une estimation plus robuste de l'importance relative de chaque variable pour
                        prédire l'indice de bonheur.
                        """)
                    elif model_choice == "XGBoost":
                        st.info("""
                        💡 Dans XGBoost, l'importance des variables est calculée en fonction de leur contribution
                        à l'amélioration du modèle à chaque boost. Plus une variable est utilisée pour faire des
                        splits importants, plus son score d'importance est élevé.
                        """)

          
    
                    # Affichage des métriques
                    st.write("### Métriques de performance")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("MAE Train", 
                                 round(mean_absolute_error(y_train, y_pred_train), 3))
                    with col2:
                        st.metric("MAE Test", 
                                 round(mean_absolute_error(y_test, y_pred_test), 3))
                    with col3:
                        st.metric("R² Train", 
                                 round(r2_score(y_train, y_pred_train), 3))
                    with col4:
                        st.metric("R² Test", 
                                 round(r2_score(y_test, y_pred_test), 3))
                        
                    st.markdown('<hr>', unsafe_allow_html=True)
                    
                    # Visualisation des prédictions
                    st.write("### Visualisation des prédictions")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    plt.scatter(y_pred_test, y_test, c='green', alpha=0.5)
                    plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), 
                            color='red', linestyle='--')
                    plt.xlabel("Prédictions")
                    plt.ylabel("Valeurs réelles")
                    plt.title(f'Life Ladder: Prédictions vs Valeurs réelles - {model_choice}')
                    st.pyplot(fig)
    
                    # Importance des variables et visualisation spécifique au modèle
                    if model_choice in ["Decision Tree", "Random Forest", "XGBoost"]:
                        main_columns = [
                            'Log GDP per capita', 'Social support', 'Healthy life expectancy at birth',
                            'Freedom to make life choices', 'Generosity', 'Perceptions of corruption',
                            'Positive affect', 'Negative affect', 'Economic Freedom', 'Personal Freedom',
                            'year'
                        ]
                        st.write("### Importance des variables")
                        importances = pd.DataFrame(
                            model.feature_importances_,
                            index=Xtrain.columns,
                            columns=["Importance"]
                        )
                        importances = importances.loc[main_columns].sort_values("Importance", ascending=True)  # Ascending=True pour avoir les plus importantes en haut
                        
                        # On garde les 4 variables les plus importantes
                        top_importances = importances.tail(4)
                        
                        # Création du graphique interactif avec Plotly
                        fig = px.bar(
                            top_importances,
                            x='Importance',
                            y=top_importances.index,
                            orientation='h',  
                            title=f"Top 4 variables importantes - {model_choice}",
                            color='Importance',  # Coloration selon l'importance
                            color_continuous_scale='viridis'  
                        )
                        
                        # Personnalisation du graphique
                        fig.update_layout(
                            height=600,
                            yaxis_title="Variables",
                            xaxis_title="Importance relative",
                            showlegend=False,
                            hoverlabel=dict(
                                bgcolor="white",
                                font_size=12,
                                font_family="Arial"
                            ),
                            # Ajout d'annotations pour le pourcentage
                            annotations=[
                                dict(
                                    x=value,
                                    y=idx,
                                    text=f"{value:.1%}",
                                    showarrow=False,
                                    xanchor='left',
                                    xshift=10,
                                    font=dict(size=10)
                                ) for idx, value in zip(top_importances.index, top_importances['Importance'])
                            ]
                        )
                        
                        # Personnalisation des barres
                        fig.update_traces(
                            marker_line_color='black',
                            marker_line_width=1,
                            opacity=0.8,
                            hovertemplate="<b>%{y}</b><br>" +
                                          "Importance: %{x:.1%}<br>" +
                                          "<extra></extra>"  # Nous Supprimons le texte secondaire au survol
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                      

                        ##
                        # Visualisation spécifique au type de modèle
                        if model_choice == "Decision Tree":
                            # Code de visualisation de l'arbre
                            st.write("### Visualisation de l'arbre de décision")
                            dot_data = export_graphviz(
                                model,
                                feature_names=list(Xtrain.columns),
                                filled=True,
                                rounded=True,
                                special_characters=True,
                                max_depth=3
                            )
                           
                            
                            # Visualisation matplotlib de l'arbre
                            fig, ax = plt.subplots(figsize=(15, 10))
                            tree.plot_tree(model, 
                                         feature_names=list(Xtrain.columns),
                                         filled=True,
                                         rounded=True,
                                         fontsize=10,
                                         max_depth=3)
                            st.pyplot(fig)
                            
                            # Interprétation une seule fois ici
                            st.write("### Interprétation de l'arbre de décision")
                            st.write("""
                            L'arbre de décision ci-dessus peut être interprété comme suit:
                            - Chaque nœud représente une décision basée sur une variable
                            - Les valeurs dans les nœuds indiquent:
                                - samples: nombre d'échantillons dans le nœud
                                - value: la valeur moyenne de l'indice de bonheur
                                - mse: l'erreur quadratique moyenne
                            - Plus la couleur est foncée, plus la valeur prédite est élevée
                            """)
                        
                        elif model_choice == "Random Forest":
                            st.write("### Visualisation d'un arbre du Random Forest")
                            
                            
                            st.markdown("""
                                <style>
                                    .textorange {
                                        color: #ec5a53;
                                        font-weight: bold;
                                    }
                                </style>
                            """, unsafe_allow_html=True)
                            
                            
                            st.markdown("Le Random Forest est composé de plusieurs arbres de décision. Voici la visualisation d'un des arbres de la forêt (<span class='textorange'>le premier</span>).", unsafe_allow_html=True)


                            
                           
                            
                            fig, ax = plt.subplots(figsize=(15, 10))
                            tree.plot_tree(model.estimators_[0],
                                         feature_names=list(Xtrain.columns),
                                         filled=True,
                                         rounded=True,
                                         fontsize=10,
                                         max_depth=3)
                            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Une erreur s'est produite lors de la modélisation: {str(e)}")

    
        # ###########           

    
    elif menu == "Analyse Machine Learning":
       st.image("https://myfiteo.app/src_img_happy/img10.png", use_column_width=True)
       st.markdown("<h1>Analyse Machine Learning</h1>", unsafe_allow_html=True)
       #st.markdown("Voici notre rapport sur la modélisation et la prédiction", unsafe_allow_html=True)
       st.markdown('<hr>', unsafe_allow_html=True)
       
      
       
       st.markdown("""<br><p align="justify">Au moment de l'entrainement des modèles on remarque l'existence d'un surapprentissage. Pour limiter ce problème et optimiser le modèle afin qu'il soit le plus performant, nous avons utilisé un algorithme permettant d'optimiser les paramètres de chaque modèle.
Il convient alors de ré-entrainer les modèles avec ces paramètres optimisés et afficher les métriques MAE et R² pour chaque modèle.</p>""",unsafe_allow_html=True)
       st.markdown("""<div style="display: flex; justify-content: center;"> <img src="https://myfiteo.app/src_img_happy/g2.png"/></div><br>""",unsafe_allow_html=True)
       st.info("""
         💡 Il est pratique de pouvoir visuellement analyser les métriques de chaque modèle.
         - **MAE :**
         XGBoost a une bonne performance en test, mais présente un risque de surajustement.
         Random Forest semble équilibré, mais ses erreurs pourraient être réduites avec des ajustements d'autres hyperparamètres.
         Régression linéaire montre une bonne robustesse, mais pourrait ne pas capturer la complexité des données.
         Decision Tree est clairement surajusté aux données d'entraînement.
        
         - **R² :**
         Random Forest est plus robuste avec un meilleur équilibre entre les ensembles train et test.
         XGBoost est très performant mais présente un léger risque de surajustement.
         Regression linéaire est un modèle robuste et simple.
         Decision Tree est fortement surajusté, ce qui en fait un mauvais choix sans régularisation ou assemblage.
        
        """)
               #, unsafe_allow_html=True)

        
       st.markdown("""<div style="display: flex; justify-content: center;"> <img src="https://myfiteo.app/src_img_happy/g3.png"/></div>""",unsafe_allow_html=True)
       st.markdown("""<div style="display: flex; justify-content: center;"> <img src="https://myfiteo.app/src_img_happy/g4.png"/></div>""",unsafe_allow_html=True)

       st.write("### Interprétation des résultats")
       st.info("""
       📊 Analyse des performances :
       - **Régression linéaire** présente des métriques très proches entre entraînement et test. Ce qui est idéal pour éviter le surapprentissage. Ce modèle est donc à privilégier.
       - **Random Forest** offre des performances légèrement supérieures en R2 et en MAE tout en restant équilibré.
       - **XGBoost** sera privilégié si la priorité est la performance brute et que le surapprentissage peut être atténué par ajustement d'autres hyperparamètres.
       - **Decision Tree** affiche une grande différence entre les mesures d'entraînement et de test ce qui montre un fort surapprentissage, le rendant le moins fiable. 
       """)
        
       st.markdown("""<br>On peut rapidement remarquer que <b>le PIB</b> est la variable la plus importante du set d'après la Heatmap présentée et les features importances des modèles entrainés. <br>Les autres variables ont des résultats très proches entre elles, ce qui explique cet ordre d'importance différent.""",unsafe_allow_html=True)
       st.markdown("""<div style="display: flex; justify-content: center;"> <img src="https://myfiteo.app/src_img_happy/g1.png"/></div>""",unsafe_allow_html=True)

       

    if menu == "Conclusion":
                st.image("https://myfiteo.app/src_img_happy/img11.png", use_column_width=True)
                st.markdown("""<h1>Conclusion</h1><hr> """, unsafe_allow_html=True)
                #st.markdown('<hr>', unsafe_allow_html=True)


        
                st.markdown("""
                <style>
                    .textorange {
                        color: #ec5a53;
                        font-weight: regular;
                    }
                    .quote {
                        font-style: italic;
                        font-size: 1.1em;
                    }
                </style>
                <p align="justify">
                    Cette analyse met en évidence les principaux facteurs influençant le bonheur à travers le monde et en Europe. 
                    Les modèles prédictifs ont permis de mieux comprendre l'impact des variables sociales, économiques et politiques 
                    sur le bien-être global. <br>
                    Au travers de la DataViz et de la modélisation, il a été mis en évidence que 
                    la richesse par habitant (PIB) d'un pays est le facteur influençant le plus l'indice de bonheur. 
                    <br>Notre analyse pourrait encore être enrichie avec d'autres aspects comme le bien-être au travail, le genre, l'âge des participants, etc. 
                    D'ailleurs, les données de 2024 qui viennent d'être publiées intègrent une notion de génération.<br>
                    A cet effet, nos recommandations s'adressent aux entreprises qui doivent s'engager dans des pratiques de
                    Responsabilité Sociale des Entreprises (RSE). Celles-ci favorisent le bien-être des communautés locales, 
                    contribuant ainsi à améliorer l'aspect positif dans la société, 
                    même dans un contexte d'augmentation du PIB.
                </p>
                
                Comme le dit si bien Emmanuel Kant: 
                <p class="quote">
                    💡 <span class="textorange">"Le concept de bonheur est un concept si indéterminé que, malgré le désir qu'à tout homme d'arriver à être heureux, 
                    personne ne peut jamais dire ce que réellement il veut et il désire. 
                    Le bonheur est un idéal non de la raison mais de l'imagination."</span> 
                </p>
                """, unsafe_allow_html=True)


    
    else:
            st.stop()  
