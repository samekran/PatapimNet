#Import libraries

import os

from langchain_ibm import WatsonxEmbeddings, WatsonxLLM
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain.tools.render import render_text_description_and_args
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
import warnings

# Suppress the USER_AGENT warning
warnings.filterwarnings('ignore', message='USER_AGENT environment variable not set')

# Set a default USER_AGENT
os.environ["USER_AGENT"] = "PlantCare/1.0"

#Set up keys and api endpoints
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": "XHfmHfEIismZnFOi-vviiM-IHbxwnbsaFbM0DA12qhuT",
    "model" : "ibm/granite-3-2-8b-instruct",
    "project_id": "02464812-b861-454c-b58c-2cb85043d848",
}


# Load in model and parameters
llm = WatsonxLLM(
    model_id=credentials["model"],
    url =credentials["url"],
    apikey=credentials["apikey"],
    project_id=credentials["project_id"],
    params={
        GenParams.DECODING_METHOD: "greedy",
        GenParams.TEMPERATURE: 0,
        GenParams.MIN_NEW_TOKENS: 5,
        GenParams.MAX_NEW_TOKENS: 250,
        GenParams.STOP_SEQUENCES: ["Human:", "Observation"]
    },   
)


urls = [
  #Apple scab
  "https://www.rhs.org.uk/disease/apple-and-pear-scab",
  "https://www.davey.com/insect-disease-resource-center/apple-scab/",
  "https://mortonarb.org/plant-and-protect/tree-plant-care/plant-care-resources/apple-scab/",
  "https://en.wikipedia.org/wiki/Apple_scab",
  "https://extension.psu.edu/apple-disease-apple-scab",

  #Black Rot
  "https://www.umass.edu/agriculture-food-environment/vegetable/fact-sheets/brassicas-black-rot",
  "https://www.canr.msu.edu/ipm/diseases/black_rot?language_id=",

  #Cedar apple rust
  "https://content.ces.ncsu.edu/cedar-apple-rusts",
  "https://www.fs.usda.gov/wildflowers/plant-of-the-week/gymnosporangium_juniperi-virginianae.shtml",
  "https://ohioline.osu.edu/factsheet/plpath-tree-10",

  #Powdery Mildew
  "https://edis.ifas.ufl.edu/publication/PP267",
  "https://www.business.qld.gov.au/industries/farms-fishing-forestry/agriculture/biosecurity/plants/diseases/horticultural/powdery-mildew",
  "https://hgic.clemson.edu/factsheet/powdery-mildew/",
  "https://extension.colostate.edu/topic-areas/yard-garden/powdery-mildews-2-902/",

  #Cercospora leaf spot/gray leaf spot
  "https://edis.ifas.ufl.edu/publication/PP267",
  "https://content.ces.ncsu.edu/cercospora-leaf-spot-frogeye-leaf-spot-on-pepper",
  "https://www.umass.edu/agriculture-food-environment/greenhouse-floriculture/fact-sheets/leaf-spot-diseases-of-floricultural-crops-caused-by-fungi-bacteria",

  #Common rust
  "https://cals.cornell.edu/field-crops/corn/diseases-corn/common-rust",
  "https://extension.umn.edu/corn-pest-management/common-rust-corn",
  
  #Northern leaf blight
  "https://www.cropscience.bayer.us/articles/bayer/managing-northern-corn-leaf-blight",
  "https://extension.umn.edu/corn-pest-management/northern-corn-leaf-blight",
  "https://ohioline.osu.edu/factsheet/plpath-cer-10",

  #Esca (black measles)
  "https://www.farmprogress.com/grapes/lime-sulfur-seen-as-aid-for-black-measles-of-grapes",
  "https://grapes.extension.org/grapevine-measles/",

  #Leaf blight (isariopsis leaf spot)
  "https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/leaf-blight",
  "https://eos.com/blog/leaf-blight/",


  #Bacterial spot
  "http://extension.msstate.edu/publications/bacterial-speck-and-bacterial-spot-tomatoes",
  "https://www.ontario.ca/page/bacterial-diseases-tomato-bacterial-spot-bacterial-speck-and-bacterial-canker",

  #Early blight
  "https://www.gardeningknowhow.com/edible/vegetables/potato/potato-early-blight-treatment.htm",
  "https://extension.umn.edu/disease-management/early-blight-tomato-and-potato",
  "https://content.ces.ncsu.edu/early-blight-of-tomato",

  #Late blight
  "https://www.gardentech.com/disease/late-blight",
  "https://extension.umn.edu/disease-management/late-blight",
  "https://www.ndsu.edu/agriculture/extension/publications/late-blight-potato",

  #Leaf scorch
  "https://www.bartlett.com/resources/diseases/bacterial-leaf-scorch",
  "https://www.pubs.ext.vt.edu/3001/3001-1433/3001-1433.html",
  "https://www.gardenia.net/disease/leaf-scorch",

  #Mold
  "https://ipm.ucanr.edu/PMG/GARDEN/PLANTS/DISEASES/botrytis.html",
  "https://extension.umn.edu/plant-diseases/gray-mold-flower-garden",
  "https://hort.extension.wisc.edu/articles/white-mold/",

  #Spider mites (two spotted spider mite)
  "https://extension.usu.edu/planthealth/ipm/notes_ag/hemp-spider-mites",
  "https://www.udel.edu/academics/colleges/canr/cooperative-extension/fact-sheets/two-spotted-spider-mites/",

  #Target spot
  "https://plantpath.ifas.ufl.edu/u-scout/tomato/target-spot.html",
  "https://www.growingproduce.com/vegetables/take-the-right-aim-to-tame-target-spot-of-tomato/",

  #Tomato mosaic virus
  "https://www.creative-diagnostics.com/blog/index.php/what-is-tomato-mosaic-virus/",
  "https://www.myefco.com/int/green-ideas/mosaico-del-pomodoro-cose-e-come-trattarlo/",

  #Tomato yellow leaf curl virus
  "https://ipm.ucanr.edu/agriculture/tomato/tomato-yellow-leaf-curl/#gsc.tab=0",
  "https://agriculture.vic.gov.au/biosecurity/plant-diseases/vegetable-diseases/tomato-yellow-leaf-curl-virus",
  "https://content.ces.ncsu.edu/tomato-yellow-leaf-curl-virus"

]

def initialize_rag():
    # Load and process documents
    requests_kwargs = {
        'headers': {
            'User-Agent': 'PlantCare/1.0 (Plant Disease Detection Application)'
        }
    }
    docs = [WebBaseLoader(url, requests_kwargs=requests_kwargs).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    
    # Initialize embeddings
    embeddings = WatsonxEmbeddings(
        model_id=EmbeddingTypes.IBM_SLATE_30M_ENG.value,
        url=credentials["url"],
        apikey=credentials["apikey"],
        project_id=credentials["project_id"],
    )
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="plant-care-chroma",
        embedding=embeddings,
    )
    
    return vectorstore.as_retriever()

def get_care_recommendations(species: str, condition: str) -> str:
    """Get care recommendations for a plant based on species and condition."""
    retriever = initialize_rag()
    
    # Create query
    query = f"What are the care recommendations for a {species} plant with {condition}?"
    
    # Get relevant documents
    docs = retriever.invoke(query)
    
    # Create context from documents
    context = "\n".join([doc.page_content for doc in docs])
    
    # Create prompt template
    template = """Based on the following context, provide care recommendations for a {species} plant with {condition} condition:

    Context: {context}
    
    Provide a concise, practical recommendation focusing on immediate actions and preventive measures.
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Get recommendation
    chain = prompt | llm
    result = chain.invoke({
        "species": species,
        "condition": condition,
        "context": context
    })
    
    return result