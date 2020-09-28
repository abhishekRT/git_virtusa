from django.contrib import admin
from . models import Beneficiary, Inpatient, Outpatient, BeneficiaryTest, InpatientTest, OutpatientTest, Train, Test


admin.site.register(Beneficiary)
admin.site.register(Inpatient)
admin.site.register(Outpatient)
admin.site.register(BeneficiaryTest)
admin.site.register(InpatientTest)
admin.site.register(OutpatientTest)
admin.site.register(Train)
admin.site.register(Test)
