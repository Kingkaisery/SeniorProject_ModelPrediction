class data_input:
	'''This is a class for interacting directly with the data provided
	by the competition csv file.

	Input is a parsed row from the csv module'''
	def __init__(self, parsedRow):
		values = [string.strip().lower() for string in parsedRow]
		categories = ["title", "level","organization", "joblocation", "education",
						"experience", "employmentType", "industry", "jobfunction",
						"SalaryMid"]
		self.data = dict(zip(categories, values))
		
