# Main application models (Flask-SQLAlchemy)
from .db import db, Unloading, UnloadingItem, Loading, LoadingItem, Product, Warehouse, OperationType, VisionServer

# External database models (SQLAlchemy)
from .external_db import (
    Supplier, Customer, MaterialType, RawMaterial, Unit, Truck,
    Shipments, Purchases, Sales, Products as ExternalProducts, 
    Consumption, ConsumptionProfile, Alert, Invoice, Havaleh, HavalehItem,
    AnbarSangin, AnbarSalonTolid, AnbarParvandeh, AnbarMuhvatehKardan,
    AnbarKoochak, AnbarKhamirKordan, AnbarKhamirGhadim, AnbarAkhal,
    AnbarMuhavatehHomayoun, AnbarPAK, WeightAdjustmentLog
) 