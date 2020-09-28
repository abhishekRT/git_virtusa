from django.http import HttpResponse
from data.models import Inpatient, Outpatient, Beneficiary, Test, Train, InpatientTest, OutpatientTest, BeneficiaryTest
from data.code import fraud, Claim_level
from django.shortcuts import render
def home(request):
	return render(request,'home.html')
def aboutus(request):
	return render(request,'aboutus.html')
def test(request):
	b = Test.objects.all()[:50]
	i = InpatientTest.objects.all()[:50]
	o = OutpatientTest.objects.all()[:50]
	return render(request,'test.html', {'b':b, 'i':i, 'o':o})
def statistics(request):
	return render(request,'statistics.html')
def theme(request):
	return render(request,'theme.html')
def detect(request):
	pidlist = []
	invalidpid = False
	invalidcid = False
	cid = ''
	pid = ''
	cidfraud = ''
	age = ''
	piddown = False
	frauds = {'Unbundling': 0, 'False Billing': 0}
	pidresult = ''
	cidresult = ''
	ciddown = False
	if 'pid' in request.GET:
		pid = request.GET['pid']
		pidlist, frauds = fraud(request.GET['pid'])
		if not frauds:
			frauds = {'Unbundling': 0, 'False Billing': 0}
			invalidpid = True
		print(pidresult)
		piddown = True
	else:
		print('nothing1')
	if 'cid' in request.GET:
		cid = request.GET['cid']
		age, cidfraud = Claim_level(request.GET['cid'])
		if age == 0 and cidfraud == 0:
			invalidcid = True
		print(age, 'age')
		try:
			age = InpatientTest.objects.filter(ClaimID = request.GET['cid'])[0].BeneID
			age = 2009 - int(str(BeneficiaryTest.objects.filter(BeneID=age)[0].DOB).split('-')[0])
		except:
			pass
		try:
			print('came here')
			age = OutpatientTest.objects.filter(ClaimID = request.GET['cid'])[0].BeneID
			age = 2009 - int(str(BeneficiaryTest.objects.filter(BeneID=age)[0].DOB).split('-')[0])
			print('came')
		except:
			pass
		ciddown = True
	else:  
		print('nothing2')
	print(piddown, ciddown)
	print(cid)
	print(frauds)
	return render(request, 'detect.html', {'piddown': piddown, 'ciddown': ciddown, 'cidresult':cidresult, 'unbundlingcount':frauds['Unbundling'], 'falsebillingcount':frauds['False Billing'], 'fraudcount':frauds['Unbundling']+frauds['False Billing'], 'age':age, 'cidfraud': cidfraud, 'pid':pid, 'cid':cid, 'invalidpid':invalidpid, 'invalidcid':invalidcid, 'pidlist':len(pidlist)})
def display(request):
	pid = request.GET['providerid']
	# ageList = [0, 0, 0, 0, 0]
	# maleIn = 0
	# femaleIn = 0
	# maleOut = 0
	# femaleOut = 0
	# maleInpatientCount = InpatientTest.objects.filter(Provider=pid)
	# for i in InpatientTest.objects.filter(Provider=pid):
	# 	gender = list(BeneficiaryTest.objects.filter(BeneID=i.BeneID))[0].Gender
	# 	age = list(BeneficiaryTest.objects.filter(BeneID=i.BeneID))[0].DOB
	# 	age = 2009-int(str(age).split('-')[0])
	# 	ageList[age//20] += 1
	# 	if gender == 1:
	# 		maleIn += 1 
	# 	else:
	# 		femaleIn += 1 
	# for i in OutpatientTest.objects.filter(Provider=pid):
	# 	gender = list(BeneficiaryTest.objects.filter(BeneID=i.BeneID))[0].Gender
	# 	age = list(BeneficiaryTest.objects.filter(BeneID=i.BeneID))[0].DOB
	# 	age = 2009-int(str(age).split('-')[0]) 
	# 	if age >= 0 and age <= 20:
	# 		ageList[0] += 1 
	# 	elif age >= 21 and age <= 40:
	# 		ageList[1] += 1 
	# 	elif age >= 41 and age <= 60:
	# 		ageList[2] += 1 
	# 	elif age >= 61 and age <= 80:
	# 		ageList[3] += 1 
	# 	else:
	# 		ageList[4] += 1 
	# 	if gender == 1:
	# 		maleOut += 1 
	# 	else:
	# 		femaleOut += 1 	
	# text = fraud(pid)
	# return render(request, 'display.html', {'pid':pid, 'maleIn':maleIn, 'maleOut':maleOut, 'femaleIn':femaleIn, 'femaleOut':femaleOut, 'age0_20':ageList[0], 'age21_40':ageList[1], 'age41_60':ageList[2], 'age61_80':ageList[3], 'age81_':ageList[4], 'in': maleIn + femaleIn, 'out': maleOut+femaleOut, 'text':text})
	return render(request, 'display.html')