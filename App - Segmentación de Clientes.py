#Librerías
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Configuración
st.set_page_config(
    page_title='Segmentación de clientes',
    page_icon='👥 ',
    layout='wide'
)

#Carga del preprocesador y el modelo
@st.cache_resource
def cargar_modelos():
    preprocessor = joblib.load('/Preprocesamiento_Segmentacion_Clientes.pkl')
    kmeans = joblib.load('/Modelo_Segmentacion_Clientes.pkl')

    return preprocessor, kmeans

#Carga de datos
@st.cache_data
def cargar_datos():
    return pd.read_csv('Datos Procesados - Segmentación de Clientes.csv')

#Preprocesador, modelos y datos
preprocessor, kmeans = cargar_modelos()
data = cargar_datos()

#Panel Lateral
st.sidebar.title('👥 Segmentación de clientes')

#Secciones
pagina = st.sidebar.radio(
    'Seleccione una sección',
    [
        'Información del proyecto',
        'Exploración de segmentos',
        'Clasificar cliente nuevo'
    ]
)

#Información del proyecto
if pagina == 'Información del proyecto':

    st.title('👥 Segmentación de clientes')

    st.write("""
    Esta aplicación contiene los resultados de un modelo de segmentación
    de clientes desarrollado mediante técnicas de aprendizaje no supervisado.

    El objetivo es identificar grupos de clientes con comportamientos de compra similares
    y sugerir estrategias comerciales para cada segmento.
    """)
    
    st.divider()

    #Información principal
    col1, col2, col3, col4 = st.columns(4)

    #Modelo
    col1.metric(
        label='Modelo:',
        value='K-Means'
    )

    #Segmentos
    col2.metric(
        label='Segmentos:',
        value='2'
    )

    #Silhouette Score
    col3.metric(
        label='Silhouette Score:',
        value='0.27'
    )

    #Total de clientes
    col4.metric(
        label='Clientes analizados:',
        value=f'{len(data):,}'
    )

    st.divider()

    st.subheader('Perfiles de los clientes')

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        ### 💎 Clientes de alto valor

        **994 clientes**

        Características:

        - Mayores ingresos.
        - Mayor gasto total.
        - Mayor frecuencia de compra.
        - Menor cantidad promedio de hijos.
        - Mayor aceptación de ofertas.
        - Comportamiento de compra multicanal.
        """)

    with col2:

        st.markdown("""
        ### 🛒 Clientes de bajo consumo

        **1,243 clientes**

        Características:

        - Menores ingresos.
        - Menor gasto total.
        - Menor frecuencia de compra.
        - Mayor cantidad promedio de hijos.
        - Menor aceptación de ofertas
        - Mayor concentración de compras en tienda y sitio web.
        """)
    
    st.info("""
    Estos segmentos representan perfiles generales de comportamiento.

    El Silhouette Score obtenido indica una separación moderada,
    por lo que pueden existir clientes con características intermedias
    entre ambos grupos.
    """)


#Exploración de segmentos
elif pagina == 'Exploración de segmentos':
    st.title('📊 Exploración de segmentos')
    st.write("""
    En esta sección se presentan las principales características
    de los segmentos identificados por el modelo K-Means.
    """)
    st.divider()


    #Distribución de clientes por segmento
    st.subheader('Distribución de clientes')
    
    conteo_clusters = (
        data['Cluster_KMeans']
        .value_counts()
        .sort_index()
    )

    col1, col2 = st.columns(2)
    col1.metric(
        'Clientes de alto valor',
        f'{conteo_clusters[0]:,}'
    )
    col2.metric(
        'Clientes de bajo consumo',
        f'{conteo_clusters[1]:,}'
    )

    #Gráfico de barras
    fig, ax = plt.subplots(figsize=(7,5))
    sns.countplot(
        data = data,
        x = 'Cluster_KMeans',
        ax = ax,
        color = sns.color_palette('YlOrBr')[1]
    )
    #Etiquetas
    ax.set_title('Distribución de clientes por segmento')
    ax.set_xlabel('Segmento')
    ax.set_ylabel('Número de clientes')
    ax.set_xticks([0, 1])
    ax.set_xticklabels([
        'Alto valor',
        'Bajo consumo'
    ])
    
    #Etiquetas de las barras con porcentajes
    total = len(data)
    heights = [p.get_height() for p in ax.patches]
    labels = [f'{(h/total)*100:.2f}%' for h in heights]
    ax.bar_label(ax.containers[0], labels = labels, padding = 1)
    
    st.pyplot(fig)
    st.divider()


    #Perfil de los segmentos
    st.subheader('Perfil de cada segmento')
    
    perfil = (data.groupby('Cluster_KMeans').agg(
                Ingreso_Promedio = ('Income', 'mean'),
                Gasto_Promedio = ('Total_gasto', 'mean'),
                Compras_Promedio = ('Total_compras', 'mean'),
                Hijos_Promedio = ('Total_hijos', 'mean'),
                Campañas_Aceptadas_Promedio = ('Campañas_aceptadas', 'mean')
            ).round(2)
        )

    perfil.index = [
        'Alto Valor',
        'Bajo consumo'
    ]

    st.dataframe(
        perfil,
        use_container_width=True
    )


    #Ingresos y gasto total por segmento
    st.subheader('Ingreso anual vs. gasto total')
    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(
        data = data[data['Income'] < 200000],
        x = 'Income',
        y = 'Total_gasto',
        hue = 'Cluster_KMeans',
        alpha = 0.65,
        ax = ax
    )

    ax.set_title('Ingresos y gastos por segmento de clientes')
    ax.set_xlabel('Ingreso anual')
    ax.set_ylabel('Gasto total')
    
    #Leyenda
    leg = ax.legend(title = 'Segmento de clientes')
    nuevas_etiquetas = ['Alto valor', 'Bajo consumo'] 
    for text, nueva_etiqueta in zip(leg.get_texts(), nuevas_etiquetas):
        text.set_text(nueva_etiqueta)

    st.pyplot(fig)
    st.divider()


    #Tabs de productos y canales de compra
    tab1, tab2 = st.tabs([
        '🛍️ Productos',
        '🏪 Canales de compra'
    ])

    #Productos
    with tab1:

        st.subheader('Preferencias de productos por segmento')
        st.write("""
        La siguiente visualización muestra qué porcentaje del gasto total dedica cada segmento
        a los diferentes tipos de productos ofrecidos.
        """)

        #Productos
        productos = [
            'MntWines',
            'MntFruits',
            'MntMeatProducts',
            'MntFishProducts',
            'MntSweetProducts',
            'MntGoldProds'
        ]

        #Gasto total
        gasto_productos = (data.groupby('Cluster_KMeans')[productos].sum())
        
        #Porcentajes
        porcentaje_productos = (gasto_productos.div(gasto_productos.sum(axis = 1), axis = 0) * 100).round(2)

        porcentaje_productos.columns = [
            'Vinos',
            'Frutas',
            'Carnes',
            'Pescados',
            'Dulces',
            'Oro'
        ]

        porcentaje_productos.index = [
            'Alto valor',
            'Bajo consumo'
        ]

        #Heatmap
        fig, ax = plt.subplots(figsize=(10,4))
        sns.heatmap(
            porcentaje_productos,
            annot=True,
            fmt='.1f',
            cmap='YlOrBr',
            ax=ax
        )

        ax.set_title('Distribución porcentual del gasto')
        ax.set_xlabel('Categoría de producto')
        ax.set_ylabel('Segmento')

        st.pyplot(fig)

        st.info("""
        🍷 El vino representa aproximadamente la mitad del gasto de ambos segmentos.

        🥩 Los clientes de mayor valor destinan una mayor proporción de su gasto a productos cárnicos.

        🏅 Los productos de oro ocupan un mayor porcentaje del gasto de los clientes de menor consumo. """)


    #Canales de compra
    with tab2:

        st.subheader('Canales de compra')
        st.write("""Esta visualización muestra cómo se distribuyen las compras
        de cada segmento entre los medios de venta: sitio web, catálogo y tienda física.""")

        canales = [
            'NumWebPurchases',
            'NumCatalogPurchases',
            'NumStorePurchases'
        ]
        
        #Total de compras
        compras_canales = (data.groupby('Cluster_KMeans')[canales].sum())

        #Porcentajes
        porcentaje_canales = (compras_canales.div(compras_canales.sum(axis=1),axis=0)* 100).round(2)
        porcentaje_canales.columns = [
            'Sitio web',
            'Catálogo',
            'Tienda física'
        ]
        porcentaje_canales.index = [
            'Mayor valor',
            'Bajo consumo'
        ]

        #Heatmap
        fig, ax = plt.subplots(figsize=(8,4))
        sns.heatmap(
            porcentaje_canales,
            annot = True,
            fmt = '.1f',
            cmap = 'YlOrBr',
            ax = ax
        )

        #Etiquetas
        ax.set_title('Distribución porcentual de compras por canal')
        ax.set_xlabel('Canal')
        ax.set_ylabel('Segmento')

        st.pyplot(fig)

        st.info("""
        🏪 La tienda física es el principal medio de compra para ambos segmentos.

        📖 Los clientes de alto valor presentan un uso mayor del catálogo que los clientes de bajo consumo.

        🌐 Los clientes de bajo consumo concentran cerca del 89% de sus compras entre la tienda física y sitio web.
        """)


#Clasificar nuevo cliente
elif pagina == 'Clasificar cliente nuevo':

    st.title('🧠 Clasificar nuevo cliente')

    st.write(""" Ingresa las características del cliente nuevo para identificar a qué segmento pertenece.""")

    st.divider()

    #Características del encoder
    encoder = (preprocessor.named_transformers_['cat'].named_steps['encoder'])
    niveles_educativos = encoder.categories_[0]
    estados_civiles = encoder.categories_[1]

    #Formulario
    with st.form('form_cliente'):
        col1, col2 = st.columns(2)
        with col1:

            #Edad
            age = st.number_input(
                'Edad (mayores de 18 años)',
                min_value = 18,
                max_value = 100,
                value = 18
            )

            #Ingresos
            income = st.number_input(
                'Ingreso anual',
                min_value = 0.0,
                value = 0.0,
                step = 500.0
            )

            #Total de hijos
            total_hijos = st.number_input(
                'Cantidad de hijos',
                min_value = 0,
                max_value = 5,
                value = 0
            )
            
            #Gasto total
            total_gasto = st.number_input(
                'Gasto total',
                min_value = 0.0,
                value = 0.0,
                step = 50.0
            )

        with col2:

            #Total de compras
            total_compras = st.number_input(
                'Cantidad total de compras',
                min_value = 0,
                value = 0
            )

            #Campañas de ofertas aceptadas
            campanas = st.number_input(
                'Campañas de ofertas aceptadas',
                min_value = 0,
                max_value = 6,
                value = 0
            )

            #Nivel educativo
            education = st.selectbox(
                'Nivel educativo',
                niveles_educativos
            )

            #Estado civil
            marital_status = st.selectbox(
                'Estado civil',
                estados_civiles
            )

        #Botón enviar
        enviar = st.form_submit_button(
            'Segmentar cliente',
            use_container_width=True
        )


    #Clasificación
    #Acción del botón
    if enviar:
        nuevo_cliente = pd.DataFrame({
            'Age': [age],
            'Income': [income],
            'Total_hijos': [total_hijos],
            'Total_gasto': [total_gasto],
            'Total_compras': [total_compras],
            'Campañas_aceptadas': [campanas],
            'Education': [education],
            'Marital_Status': [marital_status]
        })

        #Preprocesamiento
        cliente_processed = ( preprocessor.transform(nuevo_cliente))
        #Clasificación
        cluster = (kmeans.predict(cliente_processed)[0])
        st.divider()


        #Clientes de alto valor (grupo 0)
        if cluster == 0:
            
            st.success('💎 Cliente de alto valor.')
            st.write(""" Este cliente presenta características similares al segmento con mayor nivel de ingreso,
            gasto y frecuencia de compra.""")
            
            st.subheader('Características del segmento:')
            st.markdown("""
            - Mayor capacidad económica.
            - Mayor cantidad de compras.
            - Mayor gasto total.
            - Mayor receptividad histórica a campañas de ofertas.
            - Menor cantidad promedio de hijos.
            - Comportamiento de compra multicanal.
            """)

            #Estrategia comercial
            st.subheader('💡 Acciones comerciales sugeridas:')
            st.markdown("""
            - Priorizar estrategias de fidelización.
            - Promover vinos y productos cárnicos.
            - Mantener una estrategia omnicanal con el uso del sitio web, catálogo y tienda física.
            - Ofrecer recomendaciones personalizadas basadas en compras anteriores.
            """)


        #Clientes de bajo consumo (grupo 1)
        else:

            st.info('🛒 Cliente de bajo consumo.')
            st.write("""
            Este cliente presenta características similares al
            segmento con menor nivel de gasto y frecuencia de compra.
            """)
            st.subheader('Características del segmento:')
            st.markdown("""
            - Menor nivel de ingreso.
            - Menor cantidad de compras.
            - Menor gasto total.
            - Menor respuesta histórica a campañas de ofertas.
            - Mayor cantidad promedio de hijos.
            - Mayor concentración de compras en tienda y sitio web.
            """)

            st.subheader('💡 Acciones comerciales sugeridas:')
            st.markdown("""
            - Priorizar promociones mediante tienda físicay sitio web.
            - Explorar productos de oro como categoría de afinidad.
            - Aplicar estrategias de venta cruzada.
            - Utilizar promociones orientadas a aumentar el ticket promedio.
            """)


        #Aviso
        st.caption("""
        Las recomendaciones representan hipótesis comerciales
        basadas en los patrones observados en cada segmento y
        no garantizan el comportamiento futuro del cliente.
        """)