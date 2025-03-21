import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.title("Analyse des KPIs - Porsche")

@st.cache_data
def load_data():
    # Lecture du fichier avec le séparateur point-virgule
    data = pd.read_csv("dataset-porsche.csv", sep=";", on_bad_lines="skip")
    return data

df = load_data()
st.write("Aperçu du dataset :", df.head())

st.sidebar.header("Sélectionnez l'analyse KPI")
kpi = st.sidebar.radio("Choisissez une analyse :", 
                       ("1. Top modèle dans chacune des catégories",
                        "2. Normes environnementales et performances",
                        "3. Transmission et accélération",
                        "4. Puissance moteur et consommation d’huile",
                        "5. Aérodynamique, accélération et consommation",
                        "6. Evolutivité des prix"))


if kpi == "1. Top modèle dans chacune des catégories":

    required_columns = ["maximum_speed", "acceleration_0-100km/h", "co2_emissions", "price", "length", "generation"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error("Les colonnes suivantes sont manquantes dans le dataset : " + ", ".join(missing_cols))
    else:
        # Convertir en numérique les colonnes concernées
        df["maximum_speed"] = pd.to_numeric(df["maximum_speed"], errors="coerce")
        df["acceleration_0-100km/h"] = pd.to_numeric(df["acceleration_0-100km/h"], errors="coerce")
        df["co2_emissions"] = pd.to_numeric(df["co2_emissions"], errors="coerce")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["length"] = pd.to_numeric(df["length"], errors="coerce")
        
        # Conserver uniquement les lignes sans valeurs manquantes dans ces colonnes
        df_clean = df.dropna(subset=required_columns).copy()
        
        # 1. Modèle le plus rapide : vitesse maximale la plus élevée
        fastest = df_clean.loc[df_clean["maximum_speed"].idxmax()]
        
        # 2. Celui qui accélère le plus vite : temps d'accélération le plus bas (0-100 km/h)
        fastest_accel = df_clean.loc[df_clean["acceleration_0-100km/h"].idxmin()]
        
        # 3. Celui qui pollue le moins : émissions CO₂ les plus faibles
        least_polluting = df_clean.loc[df_clean["co2_emissions"].idxmin()]
        
        # 4. Le modèle le plus cher : prix le plus élevé
        most_expensive = df_clean.loc[df_clean["price"].idxmax()]
        
        # 5. Le modèle le plus grand : longueur la plus élevée
        largest = df_clean.loc[df_clean["length"].idxmax()]
        
        st.subheader("Top Modèles")
        col1, col2, col3 = st.columns(3)
        col4, col5 = st.columns(2)
        
        # Affichage des cards avec st.metric
        col1.metric("Modèle le plus rapide", f"{fastest['generation']}", f"{fastest['maximum_speed']} km/h")
        col2.metric("Accélération la plus rapide", f"{fastest_accel['generation']}", f"{fastest_accel['acceleration_0-100km/h']} s")
        col3.metric("Modèle le moins polluant", f"{least_polluting['generation']}", f"{least_polluting['co2_emissions']} g/km")
        col4.metric("Modèle le plus cher", f"{most_expensive['generation']}", f"{most_expensive['price']} €")
        col5.metric("Modèle le plus grand", f"{largest['generation']}", f"{largest['length']} m")

if kpi == "2. Normes environnementales et performances":
    st.header("KPI 1 : Indice Performance/Norme environnementale")
    st.markdown("""
    **Définition :**  
    Cet indice combine une mesure de performance (puissance ou accélération) avec les émissions (CO₂).  
    Ici, nous utilisons le ratio **power / co2_emissions**.  
    Si la colonne « power » n'est pas disponible, nous utiliserons l'inverse de l'accélération (colonne « acceleration_0-100km/h ») pour construire le ratio.
    """)
    
    # Cas avec la colonne "power"
    if 'power' in df.columns and 'co2_emissions' in df.columns:
        df_filtered = df[df['co2_emissions'] != 0].copy()
        df_filtered['Indice'] = df_filtered['power'] / df_filtered['co2_emissions']
        df_sorted = df_filtered.sort_values(by='Indice', ascending=False)
        
        # Top 10 des meilleurs ratios (tri décroissant)
        top10 = df_sorted.head(10)
        # Top 10 des pires ratios (tri croissant)
        worst10 = df_sorted.tail(10).sort_values(by='Indice', ascending=True)
        
        st.subheader("Section 1 : 10 Meilleurs modèles")
        st.write(top10[['generation', 'start_of_production', 'power', 'co2_emissions', 'Indice']])
        
        labels_top = top10.apply(lambda row: f"{row['generation']} ({row['start_of_production']})", axis=1)
        fig_top, ax_top = plt.subplots()
        # Utilisation d'une couleur bleue uniforme pour toutes les barres
        ax_top.bar(labels_top, top10['Indice'], color='blue')
        ax_top.set_xlabel("Modèle (Generation (Start_of_production))")
        ax_top.set_ylabel("Indice (power / co2_emissions)")
        ax_top.set_title("Top 10 - Meilleurs ratios")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig_top)
        
        st.subheader("Section 2 : 10 Pires modèles")
        st.write(worst10[['generation', 'start_of_production', 'power', 'co2_emissions', 'Indice']])
        
        labels_worst = worst10.apply(lambda row: f"{row['generation']} ({row['start_of_production']})", axis=1)
        fig_worst, ax_worst = plt.subplots()
        ax_worst.bar(labels_worst, worst10['Indice'], color='blue')
        ax_worst.set_xlabel("Modèle (Generation (Start_of_production))")
        ax_worst.set_ylabel("Indice (power / co2_emissions)")
        ax_worst.set_title("Top 10 - Pires ratios")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig_worst)
    
    # Cas alternatif avec la colonne "acceleration_0-100km/h"
    elif 'acceleration_0-100km/h' in df.columns and 'co2_emissions' in df.columns:
        df_filtered = df[df['co2_emissions'] != 0].copy()
        df_filtered['Indice'] = (1 / df_filtered['acceleration_0-100km/h']) / df_filtered['co2_emissions']
        df_sorted = df_filtered.sort_values(by='Indice', ascending=False)
        
        top10 = df_sorted.head(10)
        worst10 = df_sorted.tail(10).sort_values(by='Indice', ascending=True)
        
        st.subheader("Section 1 : 10 Meilleurs modèles")
        st.write(top10[['generation', 'start_of_production', 'acceleration_0-100km/h', 'co2_emissions', 'Indice']])
        
        labels_top = top10.apply(lambda row: f"{row['generation']} ({row['start_of_production']})", axis=1)
        fig_top, ax_top = plt.subplots()
        ax_top.bar(labels_top, top10['Indice'], color='blue')
        ax_top.set_xlabel("Modèle (Generation (Start_of_production))")
        ax_top.set_ylabel("Indice ((1/acceleration_0-100km/h) / co2_emissions)")
        ax_top.set_title("Top 10 - Meilleurs ratios")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig_top)
        
        st.subheader("Section 2 : 10 Pires modèles")
        st.write(worst10[['generation', 'start_of_production', 'acceleration_0-100km/h', 'co2_emissions', 'Indice']])
        
        labels_worst = worst10.apply(lambda row: f"{row['generation']} ({row['start_of_production']})", axis=1)
        fig_worst, ax_worst = plt.subplots()
        ax_worst.bar(labels_worst, worst10['Indice'], color='blue')
        ax_worst.set_xlabel("Modèle (Generation (Start_of_production))")
        ax_worst.set_ylabel("Indice ((1/acceleration_0-100km/h) / co2_emissions)")
        ax_worst.set_title("Top 10 - Pires ratios")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig_worst)
    
    else:
        st.error("Les colonnes nécessaires (power ou acceleration_0-100km/h et co2_emissions) ne sont pas disponibles dans le dataset.")

    df_filtered = df.dropna(subset=["start_of_production", "co2_emissions"]).copy()
        
        # Conversion de la colonne "start_of_production" en numérique
    df_filtered["start_of_production"] = pd.to_numeric(df_filtered["start_of_production"], errors="coerce")
    df_filtered = df_filtered.dropna(subset=["start_of_production"])
    df_filtered["start_of_production"] = df_filtered["start_of_production"].astype(int)
        
        # Calcul de la moyenne des émissions de CO₂ par année
    co2_by_year = df_filtered.groupby("start_of_production")["co2_emissions"].mean().reset_index()
    co2_by_year = co2_by_year.sort_values("start_of_production")
        
    st.subheader("Moyenne des émissions de CO₂ par année")
    st.write(co2_by_year)
        
        # Création du graphique en ligne
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(co2_by_year["start_of_production"], co2_by_year["co2_emissions"], marker="o", linestyle="-", color="green")
    ax.set_xlabel("Année de production")
    ax.set_ylabel("Emissions CO₂ moyennes")
    ax.set_title("Évolution des émissions de CO₂ par année")
    st.pyplot(fig)



if kpi == "3. Transmission et accélération":
    st.header("Transmission et Accélération")
    st.markdown("""
    Cette page présente plusieurs KPI sur l'accélération des véhicules.
    
    1. Temps d'accélération moyen par type de transmission  
    2. **Temps d'accélération moyen (0-100 km/h) par moteur**  
       - *Top 10 des moteurs les plus performants* (temps le plus bas)  
       - *Top 10 des moteurs les moins performants* (temps le plus élevé)
    3. **Vitesse maximale par génération**
        - *Top 10 voitures avec la vitesse maximale la plus élevée*
        - *Top 10 avec la vitesse maximale la plus basse*
    """)    
    # Vérifier la présence des colonnes nécessaires
    if "acceleration_0-100km/h" not in df.columns or "drive_wheel" not in df.columns:
        st.error("Les colonnes nécessaires ('acceleration_0-100km/h' et 'drive_wheel') ne sont pas disponibles.")
    else:
        # Filtrage pour supprimer les lignes avec valeurs manquantes
        df_filtered = df.dropna(subset=["acceleration_0-100km/h", "drive_wheel"])
        
        # ---------------------------------------------------------------------
        # KPI 1 : Temps d'accélération moyen par type de transmission
        st.subheader("1. Temps d'accélération moyen (0-100 km/h) par type de transmission")
        
        # Calculer la moyenne par type de transmission
        mean_acc = df_filtered.groupby("drive_wheel")["acceleration_0-100km/h"].mean().reset_index()
        st.write("Temps d'accélération moyen par type de transmission :", mean_acc)
        
        # Afficher un graphique en barres
        fig_bar, ax_bar = plt.subplots()
        ax_bar.bar(mean_acc["drive_wheel"], mean_acc["acceleration_0-100km/h"], color="blue")
        ax_bar.set_xlabel("Type de transmission")
        ax_bar.set_ylabel("Temps moyen d'accélération (0-100 km/h)")
        ax_bar.set_title("Temps d'accélération moyen par transmission")
        st.pyplot(fig_bar)
        
    
    # Nouveau KPI : Temps d'accélération moyen par moteur (Top 10 Meilleurs et Pires)
    if "acceleration_0-100km/h" not in df.columns or "engine" not in df.columns:
        st.error("Les colonnes 'acceleration_0-100km/h' et/ou 'engine' sont manquantes.")
    else:
        st.subheader("2. Temps d'accélération moyen par moteur")
        # Filtrer les données pour le KPI par moteur
        df_engine = df.dropna(subset=["acceleration_0-100km/h", "engine"]).copy()
        # Calculer la moyenne par moteur
        mean_acc_engine = df_engine.groupby("engine")["acceleration_0-100km/h"].mean().reset_index()
        
        # Trier par temps d'accélération moyen (plus bas = meilleur performance)
        mean_acc_engine_sorted = mean_acc_engine.sort_values(by="acceleration_0-100km/h", ascending=True)
        top10_best = mean_acc_engine_sorted.head(10)
        # Pour les pires, on prend les 10 dernières et on les trie par ordre décroissant (pour afficher du pire au moins pire)
        top10_worst = mean_acc_engine_sorted.tail(10).sort_values(by="acceleration_0-100km/h", ascending=False)
        
        st.write("Top 10 des moteurs les plus performants (meilleur temps d'accélération) :", top10_best)
        
        # Graphique pour les 10 meilleurs moteurs
        fig_best, ax_best = plt.subplots(figsize=(10, 6))
        labels_best = top10_best["engine"]
        ax_best.bar(labels_best, top10_best["acceleration_0-100km/h"], color="blue")
        ax_best.set_xlabel("Moteur")
        ax_best.set_ylabel("Temps moyen (0-100 km/h)")
        ax_best.set_title("Top 10 des moteurs les plus performants")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig_best)
        
        st.write("Top 10 des moteurs les moins performants (pire temps d'accélération) :", top10_worst)

        # Graphique pour les 10 pires moteurs
        fig_worst, ax_worst = plt.subplots(figsize=(10, 6))
        labels_worst = top10_worst["engine"]
        ax_worst.bar(labels_worst, top10_worst["acceleration_0-100km/h"], color="red")
        ax_worst.set_xlabel("Moteur")
        ax_worst.set_ylabel("Temps moyen (0-100 km/h)")
        ax_worst.set_title("Top 10 des moteurs les moins performants")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig_worst)

         # --- Nouveau KPI : Top 10 voitures par vitesse maximale (la plus élevée et la plus basse) ---
    if "maximum_speed" not in df.columns:
        st.error("La colonne 'maximum_speed' est manquante dans le dataset.")
    else:
        st.subheader("3. Top 10 voitures par vitesse maximale")
        df_speed = df.dropna(subset=["maximum_speed", "generation"]).copy()
        df_speed["maximum_speed"] = pd.to_numeric(df_speed["maximum_speed"], errors="coerce")
        df_speed = df_speed.dropna(subset=["maximum_speed"])
        
        # Top 10 voitures avec la vitesse maximale la plus élevée
        top10_high_speed = df_speed.sort_values(by="maximum_speed", ascending=False).head(10)
        st.write("Top 10 voitures avec la vitesse maximale la plus élevée :", top10_high_speed[["generation", "maximum_speed"]])
        fig_speed_high, ax_speed_high = plt.subplots(figsize=(10,6))
        ax_speed_high.bar(top10_high_speed["generation"], top10_high_speed["maximum_speed"], color="green")
        ax_speed_high.set_xlabel("Voiture (Generation)")
        ax_speed_high.set_ylabel("Vitesse maximale")
        ax_speed_high.set_title("Top 10 voitures avec la vitesse maximale la plus élevée")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig_speed_high)
        
        # Top 10 voitures avec la vitesse maximale la plus basse
        top10_low_speed = df_speed.sort_values(by="maximum_speed", ascending=True).head(10)
        st.write("Top 10 voitures avec la vitesse maximale la plus basse :", top10_low_speed[["generation", "maximum_speed"]])
        fig_speed_low, ax_speed_low = plt.subplots(figsize=(10,6))
        ax_speed_low.bar(top10_low_speed["generation"], top10_low_speed["maximum_speed"], color="red")
        ax_speed_low.set_xlabel("Voiture (Generation)")
        ax_speed_low.set_ylabel("Vitesse maximale")
        ax_speed_low.set_title("Top 10 voitures avec la vitesse maximale la plus basse")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig_speed_low)

if kpi == "4. Puissance moteur et consommation d’huile":
    st.header("Puissance moteur et consommation d’huile")
    st.markdown("""
    Cette page présente deux KPI :

    1. **Corrélation entre la puissance et la consommation d’huile :**  
       Un scatter plot avec une ligne de tendance, indiquant la force et la direction de la corrélation entre la puissance (`power`) et la consommation d’huile (`engine_oil_capacity`).

    2. **Indice de consommation relative :**  
       Cet indice correspond au ratio de la consommation d’huile par rapport à la puissance, calculé en divisant `engine_oil_capacity` par `power` (litres par cheval-vapeur).  
       Un indice plus bas est préférable.  
       Nous affichons ici les 10 meilleurs modèles (ceux ayant le plus faible indice).
    """)

    # Vérifier que les colonnes nécessaires existent
    if "power" not in df.columns or "engine_oil_capacity" not in df.columns:
        st.error("Les colonnes nécessaires ('power' et 'engine_oil_capacity') ne sont pas disponibles dans le dataset.")
    else:
        # Filtrage pour enlever les valeurs manquantes
        df_filtered = df.dropna(subset=["power", "engine_oil_capacity"]).copy()

        # --- KPI 1 : Corrélation entre puissance et consommation d’huile ---
        st.subheader("1. Corrélation entre puissance et consommation d’huile")
        fig_scatter, ax_scatter = plt.subplots()
        
        x = df_filtered["power"]
        y = df_filtered["engine_oil_capacity"]
        
        ax_scatter.scatter(x, y, color="blue", alpha=0.7)
        ax_scatter.set_xlabel("Puissance (power)")
        ax_scatter.set_ylabel("Consommation d’huile (engine_oil_capacity)")
        ax_scatter.set_title("Corrélation entre puissance et consommation d’huile")
        
        # Calculer et tracer la ligne de tendance (régression linéaire)
        if len(x) > 1:
            coeffs = np.polyfit(x, y, 1)
            trendline = np.poly1d(coeffs)
            x_vals = np.linspace(x.min(), x.max(), 100)
            ax_scatter.plot(x_vals, trendline(x_vals), color="red", linewidth=2, label="Ligne de tendance")
            ax_scatter.legend()
            # Afficher le coefficient de corrélation
            corr_coef = np.corrcoef(x, y)[0, 1]
            st.write(f"Coefficient de corrélation : {corr_coef:.2f}")
        st.pyplot(fig_scatter)
        
        # --- KPI 2 : Indice de consommation relative ---
        st.subheader("2. Indice de consommation relative (litres par cheval-vapeur)")
        st.markdown("""
        Cet indice est calculé en divisant la consommation d’huile (`engine_oil_capacity`) par la puissance (`power`).  
        **Un indice plus bas est meilleur.**
        """)
        # Calcul de l'indice
        df_filtered["Indice_Consommation"] = df_filtered["engine_oil_capacity"] / df_filtered["power"]
        st.write("Exemple de calcul de l'indice :", 
                 df_filtered[["engine_oil_capacity", "power", "Indice_Consommation"]].head())
        
        # Sélection des 10 meilleurs modèles (ceux avec l'indice le plus faible)
        best10 = df_filtered.sort_values(by="Indice_Consommation", ascending=True).head(10)
        
        st.write("Top 10 des modèles avec le meilleur indice de consommation relative :", 
                 best10[["generation", "start_of_production", "power", "engine_oil_capacity", "Indice_Consommation"]])
        
        # Graphique en barres pour le top 10
        # On utilise ici une couleur bleue uniforme pour toutes les barres
        labels_best = best10.apply(lambda row: f"{row['generation']} ({row['start_of_production']})", axis=1)
        fig_bar, ax_bar = plt.subplots()
        ax_bar.bar(labels_best, best10["Indice_Consommation"], color="blue")
        ax_bar.set_xlabel("Modèle (Generation (Start_of_production))")
        ax_bar.set_ylabel("Indice de consommation relative")
        ax_bar.set_title("Top 10 Meilleurs Indices (litres par cheval-vapeur)")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig_bar)

if kpi == "5. Aérodynamique, accélération et consommation":
    st.header("Aérodynamique, accélération et consommation")
    st.markdown("""
    Cette page présente trois KPI en se basant sur l'année de production (`start_of_production`) :

    1. **Évolution de la puissance par année de production :**  
       Graphique en ligne montrant l'évolution de la puissance moyenne par année.

    2. **Évolution de l'accélération (0-100 km/h) par année de production :**  
       Graphique en ligne montrant l'évolution de l'accélération moyenne par année.

    3. **Comparaison du coefficient de traînée par année de production :**  
       Diagramme en barres affichant la moyenne du coefficient de traînée pour chaque année, avec des barres d'erreur représentant l'écart-type pour une meilleure lisibilité.
    """)

    # Vérification de la présence des colonnes requises
    required_cols = ["start_of_production", "acceleration_0-100km/h", "power", "drag_coefficient"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error("Les colonnes suivantes sont manquantes dans le dataset : " + ", ".join(missing_cols))
    else:
        # Filtrage et conversion
        df_filtered = df.dropna(subset=required_cols).copy()
        df_filtered["start_of_production"] = pd.to_numeric(df_filtered["start_of_production"], errors="coerce")
        df_filtered = df_filtered.dropna(subset=["start_of_production"])

        # --- KPI 1 : Évolution de la puissance par année ---
        st.subheader("1. Évolution de la puissance par année de production")
        power_by_year = df_filtered.groupby("start_of_production")["power"].mean().reset_index()
        fig_power, ax_power = plt.subplots(figsize=(10, 6))
        ax_power.plot(power_by_year["start_of_production"], power_by_year["power"], marker="o", linestyle="-", color="blue")
        ax_power.set_xlabel("Année de production")
        ax_power.set_ylabel("Puissance moyenne")
        ax_power.set_title("Évolution de la puissance par année")
        st.pyplot(fig_power)

        # --- KPI 2 : Évolution de l'accélération par année ---
        st.subheader("2. Évolution de l'accélération (0-100 km/h) par année de production")
        accel_by_year = df_filtered.groupby("start_of_production")["acceleration_0-100km/h"].mean().reset_index()
        fig_accel, ax_accel = plt.subplots(figsize=(10, 6))
        ax_accel.plot(accel_by_year["start_of_production"], accel_by_year["acceleration_0-100km/h"], marker="o", linestyle="-", color="green")
        ax_accel.set_xlabel("Année de production")
        ax_accel.set_ylabel("Accélération moyenne (0-100 km/h)")
        ax_accel.set_title("Évolution de l'accélération par année")
        st.pyplot(fig_accel)

if kpi == "6. Evolutivité des prix":
    st.header("Évolutivité des prix")
    st.markdown("""
    Ce graphique montre l'évolution du prix en fonction des années de production.
    """)
    
    # Vérifier la présence des colonnes requises
    required_columns = ["start_of_production", "price", "generation"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        st.error("Les colonnes suivantes sont manquantes dans le dataset : " + ", ".join(missing))
    else:
        # Filtrer les données pour garder uniquement les lignes avec les colonnes requises
        df_filtered = df.dropna(subset=required_columns).copy()
        
        # Conversion de start_of_production et price en numérique
        df_filtered["start_of_production"] = pd.to_numeric(df_filtered["start_of_production"], errors="coerce")
        df_filtered = df_filtered.dropna(subset=["start_of_production"])
        df_filtered["price"] = pd.to_numeric(df_filtered["price"], errors="coerce")
        df_filtered = df_filtered.dropna(subset=["price"])
        
        # Calculer le prix moyen par année (sans distinction de génération)
        price_by_year = df_filtered.groupby("start_of_production")["price"].mean().reset_index()
        
        # Graphique en ligne pour l'évolution du prix
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(price_by_year["start_of_production"], price_by_year["price"], marker="o", linestyle="-", color="blue")
        ax.set_xlabel("Année de production")
        ax.set_ylabel("Prix")
        ax.set_title("Évolution du prix par année")
        st.pyplot(fig)
        
        # Calculer la moyenne de prix par génération pour identifier la plus chère et la moins chère
        gen_avg = df_filtered.groupby("generation")["price"].mean().reset_index()
        most_expensive = gen_avg.loc[gen_avg["price"].idxmax()]
        least_expensive = gen_avg.loc[gen_avg["price"].idxmin()]
        
        # Afficher ces informations dans deux cartes (metrics)
        col1, col2 = st.columns(2)
        col1.metric("Génération la plus chère", most_expensive["generation"], f"Prix: {most_expensive['price']:.2f}")
        col2.metric("Génération la moins chère", least_expensive["generation"], f"Prix: {least_expensive['price']:.2f}")

