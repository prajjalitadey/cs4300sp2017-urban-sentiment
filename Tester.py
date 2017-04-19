from Output import Output
from Listing import Listing

class Tester(object):

    
    def testParser():
        filename = 'placeholder enter file namer here'
        parser = Parser(filename)
        print(parser.parseairbnb())
        

    def testOutput():
        query = "great place close to wall street"
        listing = {}
        listing['Wall Street'] = [Listing(1, 'great place close to wall street', 'Wall Street')]
        listing['Fordham'] = [Listing(2, 'shitty ass neighborhood', 'Fordham')]
        listing['Eastchester'] = [Listing(3, 'Westchester', 'Eastchester')]
        listing['Manhanttan'] = [Listing(4, 'yes I know its a borough', 'Manhattan')]
        output = Output(listing)
        print(output.getNaiveImpl(query))
        
    if __name__ == '__main__':
        testOutput()
        

