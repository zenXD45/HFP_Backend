from main import app, PatientData, predict_risk
data = PatientData(**{"age":"55","gender":"M","restingBP":"120","cholesterol":"198","fastingBS":0,"chestPainType":"ASY","restingECG":"Normal","maxHR":"150","exerciseAngina":"N","oldpeak":"1.5","stSlope":"Flat"})
try:
    print(predict_risk(data))
except Exception as e:
    import traceback
    traceback.print_exc()
