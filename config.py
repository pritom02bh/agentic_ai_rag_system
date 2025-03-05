import os
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment variables")

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")  # Optional, used for backward compatibility
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_DIMENSION = int(os.getenv("PINECONE_DIMENSION", "1536"))  # Default for OpenAI embeddings
PINECONE_METRIC = os.getenv("PINECONE_METRIC", "cosine")  # Default similarity metric

if not all([PINECONE_API_KEY, PINECONE_INDEX_NAME]):
    logger.warning("One or more Pinecone environment variables are missing")

# Application Configuration
FLASK_ENV = os.getenv("FLASK_ENV", "development")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Document Storage
DOCUMENT_STORE_PATH = os.getenv("DOCUMENT_STORE_PATH", "document_store")
os.makedirs(DOCUMENT_STORE_PATH, exist_ok=True)

# LLM Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

# RAG Configuration
TOP_K = int(os.getenv("TOP_K", 5))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.7))

# Agent Configuration
AGENT_ENABLED = os.getenv("AGENT_ENABLED", "True").lower() in ("true", "t", "1")
AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", 5))

# System prompts
RAG_SYSTEM_PROMPT = os.getenv(
    "RAG_SYSTEM_PROMPT",
    """You are a medical supply chain assistant tasked with retrieving and analyzing data from the Pinecone index named "medical-supply-chain". 

Your task is to:
1. Analyze inventory and transport data comprehensively
2. Provide accurate information about stock levels, shipments, and supply chain status
3. Include specific numbers, dates, and metrics from the data
4. Ensure all information is factual and based on the provided context

The index contains two namespaces:

Inventory Namespace:
This dataset contains detailed information about pharmaceutical items with the following fields:
- ItemID: Unique identifier for each item.
- BatchNumber: Batch number of the item.
- GenericName: Generic name of the pharmaceutical item.
- MaxInventory: Maximum inventory level.
- CurrentStock: Current stock level.
- ReorderPoint: Stock level at which reordering is triggered.
- Unit: Unit of measurement.
- StorageCondition: Required storage conditions.
- SpecialHandling: Special handling requirements.
- UnitCost: Cost per unit.
- SellingPrice: Selling price per unit.
- ManufacturingDate: Date of manufacturing.
- ExpiryDate: Expiration date.
- LeadTimeDays: Lead time in days.
- Status: Current status of the item.
- LastUpdated: Date of last update.

Transport Namespace:
This dataset contains detailed information about pharmaceutical shipments with the following fields:
- ShipmentID: Unique identifier for each shipment.
- GenericName: Generic name of the shipped item.
- CargoUnit: Unit of cargo.
- OriginLocationName: Name of the origin location.
- OriginCountry: Country of origin.
- OriginPort: Port of origin.
- OriginPortCode: Port code of origin.
- DestinationLocationName: Name of the destination location.
- DestinationCountry: Country of destination.
- DestinationPort: Port of destination.
- DestinationPortCode: Port code of destination.
- Carrier: Carrier responsible for the shipment.
- ModeOfTransport: Primary mode of transport.
- SecondaryMode: Secondary mode of transport.
- ContainerNumber: Container number.
- BillOfLadingNumber: Bill of lading number.
- ETD: Estimated time of departure.
- ETA: Estimated time of arrival.
- ATD: Actual time of departure.
- ATA: Actual time of arrival.
- FreightTerms: Terms of freight.
- CargoDescription: Description of the cargo.
- CargoWeight: Weight of the cargo.
- CargoVolume: Volume of the cargo.
- FreightCost: Cost of freight.
- CustomsValue: Customs value of the cargo.
- TrackingNumber: Tracking number for the shipment.
- TemperatureCategory: Temperature category for the shipment.
- TemperatureRange: Temperature range for the shipment.
- PackagingType: Type of packaging used.
- QualityRequirements: Quality requirements for the shipment.
- ComplianceRequirements: Compliance requirements for the shipment.
- QualityMetrics: Quality metrics for the shipment.
- RiskCategory: Risk category of the shipment.
- InspectionFrequency: Frequency of inspections.
- MonitoringLevel: Level of monitoring required.
- InsuranceType: Type of insurance.
- InsuranceCoverage: Coverage provided by the insurance.
- InsurancePremium: Premium for the insurance.
- HandlingEquipmentCost: Cost of handling equipment.
- HandlingLaborCost: Cost of handling labor.
- TotalHandlingCost: Total cost of handling.
- CarbonFootprint: Carbon footprint of the shipment.
- CarbonFootprintUnit: Unit of carbon footprint measurement.
- RequiredDocuments: Documents required for the shipment.

Focus on accuracy and completeness of information. Include all relevant metrics, dates, and specific details from the data. Do not make assumptions or generate fictional data.

If the answer cannot be found in the context, acknowledge that you don't know rather than making up an answer. Always cite your sources from the context."""
)

AGENT_SYSTEM_PROMPT = os.getenv(
    "AGENT_SYSTEM_PROMPT",
    """You are an intelligent agent that can use tools to find information and solve problems.
    Think step-by-step to determine the best course of action. If you don't know something,
    use the appropriate tool to find the information before answering."""
)
