def get_neighborhood_ranking(q):
	# listing = {}
	# listing['Wall Street'] = [Listing(1, 'great place close to wall street', 'Wall Street')]
	# listing['Fordham'] = [Listing(2, 'shitty ass neighborhood', 'Fordham')]
	# listing['Eastchester'] = [Listing(3, 'Westchester', 'Eastchester')]
	# listing['Manhanttan'] = [Listing(4, 'yes I know its a borough', 'Manhattan')]

	#transcripts = read_file(11)
	query = "new york, outside of manhattan, quiet area, elderly people"
	parser = Parser()
	#parser.parseAirbnb("/Users/lyllayounes/Desktop/cs4300sp2017-urban-sentiment/jsons/test.csv")
	parser.parseAirbnb("./jsons/test.csv")
	listing = parser.getReviews()
	output = Output(listing)
	tups = output.getNaiveImpl(query)
	tmp = []
	tmp.append(["Query: "+query])
	tmp.append(["-------------------------------------------------"])

	for item in tups:
		tmp.append(item)

	return tmp