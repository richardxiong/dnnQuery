# Generating a tag utils
# By Hongyu Xiong, 04/02/2017

import math
import os
import random
import sys
import time
import re
import inspect

import editdistance as ed
import numpy as np
import scratch

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

class Config(object):
    word_dim = 100
    vocabulary_path = parentdir + '/data/vocab1000.from'
    glove_path = os.path.dirname(parentdir) + '/glove.6B/glove.6B.100d.txt'
    max_vocabulary_size = 5000

    # embedding_matrix, vocab, word_vector = scratch.generateEmbedMatrix(vocabulary_path, max_vocabulary_size, glove_path)
    # all keys for word_vector are in lowercase

    # for data augmentation
    schema_collect = [['Nation','Rank','Gold','Silver','Bronze'], #,'Total'
                      ['Name','Year_inducted','Apps','Goals'], #'Position',
                      ['State','Year_of_Election','No._of_candidates','No._of_elected','Total_no._of_seats_in_Assembly'],
                      ['Team','Years_won','County','Wins','Areas','Prices'], 
                      ['Player','Matches','Innings','50s','Games_Played','Runs','Free_Throws','Points','100s','Field_Goals'], #'Average',
                      ['Country','Masters','U.S._Open','The_Open','PGA'], #,'Total'
                      ['Nation','Name','League_Apps','League_Goals','FA_Cup_Apps','FA_Cup_Goals','Total_Apps','Total_Goals'], #'Position',
                      ['Swara','Position','Short_name','Notation','Mnemonic'], 
                      ['Year','1st_Venue','2nd_Venue','3rd_Venue','4th_Venue','5th_Venue','6th_Venue'], 
                      ['Menteri_Besar','Took_office','Left_office','Party'], 
                      ['Discipline','Amanda','Bernie','Javine_H','Julia','Michelle']  # special
                      ]
    
    schema_collect_type = [['string','ordinal','int','int','int','int'],
                      ['string','date','string','int','int'],
                      ['string','date','int','int','int'],
                      ['string','date','string','int','int','int'], 
                      ['string','int','int','int','int','int','int','int','int','int','int'], 
                      ['string','int','int','int','int'],
                      ['string','string','string','int','int','int','int','int','int'], 
                      ['string','string','string','string','string'], 
                      ['string','string','string','string','string','string','string'], 
                      ['string','date','date','string'], 
                      ['string','int','int','int','int','int']  # special
                      ]

    field2word = {
			'sum': {'value_type': 'string', 
                       'value_range': [], 
                       'query_word': ['sum','summation','total', 'combined']},
			'diff': {'value_type': 'string', 
                       'value_range': [], 
                       'query_word': ['difference']},
			'less': {'value_type': 'string', 
                       'value_range': [], 
                       'query_word': ['less']},
			'greater': {'value_type': 'string', 
                       'value_range': [], 
                       'query_word': ['more','larger']},
      'mean': {'value_type': 'string', 
                       'value_range': [], 
                       'query_word': []},
			'argmax': {'value_type': 'string', 
                       'value_range': ['maximum','max' ,'last'],#
                       'query_word': ['most','greatest','greatest','mean', 'average']}, # 'previous','before'
			'argmin': {'value_type': 'string', 
                       'value_range': [], #, 'next','after'
                       'query_word': ['least','minimum','first']},
			'Masters': {'value_type':'int',
						'query_word': ['masters']},
			'Country':{'value_type': 'string', 
                     'value_range': ['Brazil', 'Canada', 'Qatar', 'Italy', 'Peru', 'Kuwait', 'New_Zealand', 'Luxembourg', 'France',
                        'HK', 'Slovakia', 'Ireland', 'Nigeria', 'Norway', 'Argentina', 'South_Korea', 'Israel',
                        'Australia', 'Iran', 'Indonesia', 'West_Germany', 'Iceland', 'Slovenia', 'China', 'Chile',
                        'Belgium', 'Germany', 'Iraq', 'Philippines', 'Poland', 'Spain', 'Ukraine', 'Hungary',
                        'Netherlands', 'Denmark', 'Turkey', 'Finland', 'Sweden', 'Vietnam', 'Thailand', 'Switzerland',
                        'Russia', 'Pakistan', 'Romania', 'Portugal', 'Mexico', 'Egypt', 'Soviet_Union', 'Singapore',
                        'India', 'Liechtenstein', 'US', 'Czech', 'Austria', 'Yugoslavia', 'Saudi_Arabia', 'UK',
                        'Greece', 'Japan', 'Taiwan','Scotland','Mongolia','England','Kazakhstan','Nepal','Wales','Moldova',
                        'Belarus','Latvia','Armenia','United_States','Czech_Republic','Jamaica','Great_Britain','Uzbekistan',
                        'Malaysia','Uganda','Estonia','Croatia','Cuba','Morocco','Bahamas','Algeria','Cyprus','Zimbabwe',
                        'Ivory_Coast','Fiji'], 
                     'query_word': ['country','nation','countries','nations','team','who']},
      'U.S._Open': {'value_type':'int',
						'query_word': ['u.s._open']},
			'The_Open': {'value_type':'int',
						'query_word': ['the_open']},
			'PGA': {'value_type':'int',
						'query_word': ['pga']},
			'Team':{'value_type': 'string', 
                     'value_range': ['Greystones','Ballymore_Eustace','Maynooth','Ballyroan_Abbey','Fingal_Ravens','Confey',
                        'Crettyard','Wolfe_Tones','Dundalk_Gael',
                        'Brazil', 'Canada', 'Qatar', 'Italy', 'Peru', 'Kuwait', 'New_Zealand', 'Luxembourg', 'France',
                        'HK', 'Slovakia', 'Ireland', 'Nigeria', 'Norway', 'Argentina', 'South_Korea', 'Israel',
                        'Australia', 'Iran', 'Indonesia', 'West_Germany', 'Iceland', 'Slovenia', 'China', 'Chile',
                        'Belgium', 'Germany', 'Iraq', 'Philippines', 'Poland', 'Spain', 'Ukraine', 'Hungary',
                        'Netherlands', 'Denmark', 'Turkey', 'Finland', 'Sweden', 'Vietnam', 'Thailand', 'Switzerland',
                        'Russia', 'Pakistan', 'Romania', 'Portugal', 'Mexico', 'Egypt', 'Soviet_Union', 'Singapore',
                        'India', 'Liechtenstein', 'US', 'Czech', 'Austria', 'Yugoslavia', 'Saudi_Arabia', 'UK',
                        'Greece', 'Japan', 'Taiwan','Scotland','Mongolia','England','Kazakhstan','Nepal','Wales','Moldova',
                        'Belarus','Latvia','Armenia','United_States','Czech_Republic','Jamaica','Great_Britain','Uzbekistan',
                        'Malaysia','Uganda','Estonia','Croatia','Cuba','Morocco','Bahamas','Algeria','Cyprus','Zimbabwe',
                        'Ivory_Coast','Fiji'], 
                     'query_word': ['team','nation','country','who']},
      'County':{'value_type': 'string', 
                     'value_range': ['Wicklow','Kildare','Laois','Dublin','Meath','Louth'], 
                     'query_word': ['county','counties','who']},
      'Years': {'value_type':'date',
						'query_word': ['years','year','time','times','when']}, #'above','latest', 'below','after','before','previous','next'
			'Years_won': {'value_type':'date',
            'query_word': ['years','year','time','times','when']}, #'above','latest', 'below','after','before','previous','next'
      'Wins': {'value_type':'int',
						'query_word': []},
			'Areas': {'value_type':'int',
						'query_word': ['area','areas']},
			'Prices': {'value_type':'int',
						'query_word': ['prices', 'price']},
      'Swara': {'value_type': 'string', 
                     'value_range': ['Shadja','Shuddha_Rishabha','Chatushruti_Rishabha','Shuddha_Gandhara','Shatshruti_Rishabha',
                     'Sadharana_Gandhara','Antara_Gandhara','Shuddha_Madhyama','Prati_Madhyama','Panchama','Shuddha_Dhaivata',
                     'Chatushruti_Dhaivata','Shuddha_Nishada','Shatshruti_Dhaivata','Kaisiki_Nishada','Kakali_Nishada'], 
                     'query_word': ['swara','player','who']},
      'Short_name': {'value_type': 'string', 
                     'value_range': ['Pa','Sa','Ga','Gu','Gi','Ra','Ri','Ru','Ma','Mi','Dha','Dhi','Dhu','Na','Ni','Nu'], 
                     'query_word': ['short_name']},
      'Notation': {'value_type': 'string', 
                     'value_range': ['S','R1','R2','R3','G1','G2','G3','M1','M2','M3','D1','D2','D3','N1','N2','N3'], 
                     'query_word': ['notation']},
      'Mnemonic': {'value_type': 'string', 
                     'value_range': ['pa','sa','ga','gi','gu','ra','ri','ru','gi','gu','ma','mi','dha','dhi','dhu','na','ni','nu'], 
                     'query_word': ['mnemonic']},
			'Player': {'value_type': 'string', 
                     'value_range': ['Herbie_Hewett','Lionel_Palairet','Bill_Roe','George_Nichols','John_Challen','Ted_Tyler',
                     'Crescens_Robinson','Albert_Clapp','John_Felmley','Gordon_Otto','Ernest_McKay','George_Halas','Ralf_Woods',
                     'Ray_Woods','Clyde_Alwood'], 
                     'query_word': ['player','who']},
			'Matches': {'value_type':'int',
						'query_word': ['matches','match']},
			'Innings': {'value_type':'int',
						'query_word': ['innings']},
			'Runs': {'value_type':'int',
						'query_word': ['runs','run']},
			'Average': {'value_type':'int',
						'query_word': ['average']},
			'100s': {'value_type':'int',
						'query_word': ['100s']},
			'50s': {'value_type':'int',
						'query_word': ['50s']},
			'Games_Played': {'value_type':'int',
						'query_word': ['games']},
			'Field_Goals': {'value_type':'int',
						'query_word': ['field', 'goals']},
			'Free_Throws': {'value_type':'int',
						'query_word': ['free', 'throws']},
			'Points': {'value_type':'int',
						'query_word': ['points']},
			'Menteri_Besar': {'value_type': 'string', 
                     'value_range': ['Jaafar_Mohamed','Mohamed_Mahbob','Abdullah_Jaafar','Mustapha_Jaafar','Abdul_Hamid_Yusof',
                                      'Ungku_Abdul_Aziz_Abdul_Majid','Onn_Jaafar','Syed_Abdul_Kadir_Mohamed','Wan_Idris_Ibrahim'], 
                     'query_word': ['menteri_besar']},
			'Party':{'value_type': 'string', 
                     'value_range': ['Conservatives','Green','Socialist_Alternative','Independent','Labour','Respect','No_party',
                     'Liberal_Democrats','British_National_Party'], 
                     'query_word': ['party']},
			'Took_office':{'value_type': 'date', 
                     #'value_range': ['January', 'Febuary', 'March', 'April','May', 'June', 'July', 'August','September', 'December', 'November', 'October'], 
                     'query_word': ['take_office']},
      'Left_office':{'value_type': 'date', 
                     #'value_range': ['January', 'Febuary', 'March', 'April','May', 'June', 'July', 'August','September', 'December', 'November', 'October'], 
                     'query_word': ['leave_office']},
			'Nation':{'value_type': 'string', 
                     'value_range': ['Brazil', 'Canada', 'Qatar', 'Italy', 'Peru', 'Kuwait', 'New_Zealand', 'Luxembourg', 'France',
                        'HK', 'Slovakia', 'Ireland', 'Nigeria', 'Norway', 'Argentina', 'South_Korea', 'Israel',
                        'Australia', 'Iran', 'Indonesia', 'West_Germany', 'Iceland', 'Slovenia', 'China', 'Chile',
                        'Belgium', 'Germany', 'Iraq', 'Philippines', 'Poland', 'Spain', 'Ukraine', 'Hungary',
                        'Netherlands', 'Denmark', 'Turkey', 'Finland', 'Sweden', 'Vietnam', 'Thailand', 'Switzerland',
                        'Russia', 'Pakistan', 'Romania', 'Portugal', 'Mexico', 'Egypt', 'Soviet_Union', 'Singapore',
                        'India', 'Liechtenstein', 'US', 'Czech', 'Austria', 'Yugoslavia', 'Saudi_Arabia', 'UK',
                        'Greece', 'Japan', 'Taiwan','Scotland','Mongolia','England','Kazakhstan','Nepal','Wales','Moldova',
                        'Belarus','Latvia','Armenia','United_States','Czech_Republic','Jamaica','Great_Britain','Uzbekistan',
                        'Malaysia','Uganda','Estonia','Croatia','Cuba','Morocco','Bahamas','Algeria','Cyprus','Zimbabwe',
                        'Ivory_Coast','Fiji'], 
                     'query_word': ['nation','country','nations','countries','team','who']},
			'Rank': {'value_type':'ordinal',
            #'value_range': ['first', 'second', 'third', '1st', '2nd', '3rd','last'],
						'query_word': ['rank', 'ranked','ranking','ranks']}, #'above', 'below','after','before','previous','next'
      'Number': {'value_type':'ordinal',
            #'value_range': ['first', 'second', 'third', '1st', '2nd', '3rd','last'],
            'query_word': ['number', '#']}, # 'above', 'below','after','before','previous','next'
      
			'Gold': {'value_type':'int',
						'query_word': ['gold']},
			'Silver': {'value_type':'int',
						'query_word': ['silver']},
			'Bronze': {'value_type':'int',
						'query_word': ['bronze']},
			'Total': {'value_type':'int',
						'query_word': ['total']},
			'Name': {'value_type':'string',
						'value_range': ['Ned_Barkas','Harry_Brough','George_Brown','Jack_Byers','Ernie_Islip','Billy_Johnston','Robert_Jones',
            'Frank_Mann','Len_Marlow','Colin_McKay','Sandy_Mutch','Stan_Pearson','George_Richardson','Charlie_Slade',
            'Billy_E._Smith','Billy_H._Smith','Clem_Stephenson','Jack_Swann','Sam_Wadsworth','Billy_Watson','Tom_Wilson',
            'James_Wood','Tommy_Mooney','Duncan_Welbourne','Luther_Blissett','John_McClelland','David_James','Ross_Jenkins',
            'Nigel_Gibbs','Les_Taylor','Tony_Coton','Ian_Bolton','Robert_Page','Ted_Davis'],
						'query_word': ['name','who','player']},
			'Position': {'value_type':'string',
						'value_range':['Goalkeeper','Defender','Midfielder','Forward','CB','DE','TE','S','QB'],
						'query_word': ['position']},
			'Year_inducted': {'value_type':'date',
						'query_word': ['year','years','time','times','when']}, #'above','latest', 'below','after','before','previous','next'
			'Apps': {'value_type':'int',
						'query_word': ['appearance','appearances','apps']},
			'Discipline': {'value_type':'string',
            'value_range': ['Hurdles', 'Cycling', 'Swimming', 'Curling', 'Archery', 'Hammer','Whitewater_Kayak'],
            'query_word': ['discipline']},
      'Amanda':{'value_type':'int',
            'query_word': ['amanda']},
      'Bernie':{'value_type':'int',
            'query_word': ['bernie']},
      'Javine_H':{'value_type':'int',
            'query_word': ['javine_h']},
      'Julia':{'value_type':'int',
            'query_word': ['julia']},
      'Michelle':{'value_type':'int',
            'query_word': ['michelle']},
      'Goals': {'value_type':'int',
						'query_word': ['goal','goals']},
			'Total_Apps': {'value_type':'int',
						'query_word': ['total_apps']},
			'Total_Goals': {'value_type':'int',
						'query_word': ['total_goals']},
			'League_Apps': {'value_type':'int',
						'query_word': ['league_apps']},
			'League_Goals': {'value_type':'int',
						'query_word': ['league_goals']},
			'FA_Cup_Apps': {'value_type':'int',
						'query_word': ['fa_cup_apps']},
			'FA_Cup_Goals': {'value_type':'int',
						'query_word': ['fa_cup_goals']},
			'State': {'value_type':'string',
						'value_range':['California','Texas','Florida','Louisiana','Bihar','Assam','Himachal_Pradesh','Manipur',
            'Chhattisgarh','Tamil_Nadu','Jammu_and_Kashmir','Karnataka','Mizoram','Kerala','Gujarat','Rajasthan','Uttarakhand',
            'Maharashtra','Madhya_Pradesh','West_Bengal','Meghalaya','Tripura','Delhi','Goa','Punjab','Puducherry',
            'Uttar_Pradesh','Odisha','Andhra_Pradesh','Haryana'],
						'query_word': ['state','county','who']},
			'No._of_elected': {'value_type':'int',
						'query_word': ['elected','number_of_elected']},
			'No._of_candidates': {'value_type':'int',
						'query_word': ['candidate', 'candidates', 'number_of_candidates']},
			'Total_no._of_seats_in_Assembly': {'value_type':'int',
						'query_word': ['seat', 'seats','number_of_seats']},
			'Year_of_Election': {'value_type':'date',
						'query_word': ['year','years','time','times','when']}, #'above','latest', 'below','after','before','previous','next'
			'Year': {'value_type':'date',
						'query_word': ['year','years','time','times','when']}, #'above','latest', 'below','after','before','previous','next'
			'1st_Venue': {'value_type': 'string', 
                       'value_range': ['Mexico_City','Changzhou','Sheffield','Veracruz','Doha','Beijing','Dubai','Nanjing',
                       'Havana','Cambridge','San_Jose','Boston','Nassau','Detroit','Hangzhou','Indianapolis','Shanghai','Austin',
                       'Washington','Caracas','Tokyo','Guanajuato','Sydney','Los_Angeles','Chicago','Seattle','Minneapolis',
                       'Qingdao','Moscow','Tijuana'], 
                       'query_word': ['city','1st_venue']}, 
      '2nd_Venue': {'value_type': 'string', 
                       'value_range': ['Mexico_City','Changzhou','Sheffield','Veracruz','Doha','Beijing','Dubai','Nanjing',
                       'Havana','Cambridge','San_Jose','Boston','Nassau','Detroit','Hangzhou','Indianapolis','Shanghai','Austin',
                       'Washington','Caracas','Tokyo','Guanajuato','Sydney','Los_Angeles','Chicago','Seattle','Minneapolis'], 
                       'query_word': ['city', '2nd_venue']}, 
      '3rd_Venue': {'value_type': 'string', 
                       'value_range': ['Mexico_City','Changzhou','Sheffield','Veracruz','Doha','Beijing','Dubai','Nanjing',
                       'Havana','Cambridge','San_Jose','Boston','Nassau','Detroit','Hangzhou','Indianapolis','Shanghai','Austin',
                       'Washington','Caracas','Tokyo','Guanajuato','Sydney','Los_Angeles','Chicago','Seattle','Minneapolis',
                       'Edinburgh','London'], 
                       'query_word': ['city', '3rd_venue']}, 
      '4th_Venue': {'value_type': 'string', 
                       'value_range': ['Tijuana','Guanajuato','Moscow','Sydney','Los_Angeles','Tokyo','Sheffield',
                       'Shanghai','Beijing','Austin','Seattle','Veracruz','Boston','Washington','Cambridge','Havana',
                       'Nassau','Indianapolis','Changzhou','Mexico_City','Caracas','Dubai','Hangzhou','Chicago'], 
                       'query_word': ['city', '4th_venue']}, 
      '5th_Venue': {'value_type': 'string', 
                       'value_range': ['Tijuana','Guanajuato','Moscow','Sydney','Los_Angeles','Tokyo','Sheffield',
                       'Shanghai','Beijing','Austin','Seattle','Veracruz','Boston','Washington','Cambridge','Havana',
                       'Nassau','Indianapolis','Changzhou','Mexico_City','Caracas','Dubai','Hangzhou','Chicago',
                       'Guadalajara','Windsor'], 
                       'query_word': ['5th_venue']}, #'city', 
      '6th_Venue': {'value_type': 'string', 
                       'value_range': ['Guadalajara','Tijuana','Guanajuato','Moscow','Sydney','Los_Angeles','Tokyo','Sheffield',
                       'Shanghai','Beijing','Austin','Seattle','Veracruz','Boston','Washington','Cambridge','Havana',
                       'Nassau','Indianapolis','Changzhou','Mexico_City','Caracas','Dubai','Hangzhou','Chicago',
                       'Monterrey'], 
                       'query_word': [ '6th_venue']}, #'city',
	 #       'host_city':{'value_type': 'string', 
   #                     'value_range': ['Amsterdam', 'Antwerp', 'Athens', 'Atlanta', 'Bangkok', 'Barcelona', 'Beijing', 'Berlin',
   #                        'Budapest', 'Buenos_Aires', 'Cairo', 'Cape_Town', 'Chamonix', 'Chicago', 'Cortina_Ampezzo',
   #                        'Dallas', 'Delhi', 'Dubai', 'Dublin', 'Florence', 'Grenoble', 'Helsinki', 'Hong_Kong',
   #                        'Innsbruck', 'Istanbul', 'Jerusalem', 'Lake_Placid', 'Las_Vegas', 'London', 'Los_Angeles',
   #                        'Madrid', 'Melbourne', 'Mexico_City', 'Milan', 'Montreal', 'Moscow', 'Mumbai', 'Munich',
   #                        'New_York', 'Oslo', 'Paris', 'Philadelphia', 'Prague', 'Rio_de_Janeiro', 'Rome', 'San_Diego',
   #                        'Sapporo', 'Sarajevo', 'Seattle', 'Seoul', 'Macau', 'Squaw_Valley', 'St._Moritz',
   #                        'Stockholm', 'Sydney', 'Tokyo', 'Toronto', 'Venice', 'Vienna', 'Warsaw'], 
   #                     'query_word': ['city', 'cities']}, 
   #      	'#_participants':{'value_type': 'int', 
   #                          'query_word': ['participate', 'participates', 'participated', 'participant', 'participants']},
   #      	'#_audience':{'value_type': 'int', 
   #                      'query_word': ['audience']},
   #      	'#_medals':{'value_type': 'int', 
   #                    'query_word': ['medal', 'medals']},
   #      	'country_size':{'value_type': 'int', 
   #                        'query_word': ['large', 'larger', 'largest']}, 
   #      	'country_gdp':{'value_type': 'int', 
   #                       'query_word': ['gdp', 'wealth', 'wealthy', 'wealthier']}, 
   #      	'country_population':{'value_type': 'int', 
   #                              'query_word': ['population', 'people']}, 
   #      	'#_duration':{'value_type': 'int', 
   #                      'query_word': ['long', 'longer', 'longest', 'duration', 'day', 'days']},
   #      	'year':{'value_type': 'int', 
   #                'query_word': ['year', 'years','when']}, 
   #      	'country':{'value_type': 'string', 
   #                   'value_range': ['Brazil', 'Canada', 'Qatar', 'Italy', 'Peru', 'Kuwait', 'New_Zealand', 'Luxembourg', 'France',
   #                      'HK', 'Slovakia', 'Ireland', 'Nigeria', 'Norway', 'Argentina', 'South_Korea', 'Israel',
   #                      'Australia', 'Iran', 'Indonesia', 'West_Germany', 'Iceland', 'Slovenia', 'China', 'Chile',
   #                      'Belgium', 'Germany', 'Iraq', 'Philippines', 'Poland', 'Spain', 'Ukraine', 'Hungary',
   #                      'Netherlands', 'Denmark', 'Turkey', 'Finland', 'Sweden', 'Vietnam', 'Thailand', 'Switzerland',
   #                      'Russia', 'Pakistan', 'Romania', 'Portugal', 'Mexico', 'Egypt', 'Soviet_Union', 'Singapore',
   #                      'India', 'Liechtenstein', 'US', 'Czech', 'Austria', 'Yugoslavia', 'Saudi_Arabia', 'UK',
   #                      'Greece', 'Japan', 'Taiwan','Scotland','Mongolia','England','Kazakhstan','Nepal'], 
   #                   'query_word': ['country','nation','countries','nations']}
      
      }

    
    geo880_dict = {'highest_elevation': {'query_word': ['highest_elevation','highest'], 'value_type': 'int', 'value_range': []}, 
                'lowest_elevation': {'query_word': ['lowest_elevation','elevation','elevations','lowest'], 'value_type': 'int', 'value_range': []}, 
                'height': {'query_word': ['height','high','highest'], 'value_type': 'int', 'value_range': []}, 
                'population_density': {'query_word': ['population_density','density'], 'value_type': 'int', 'value_range': []}, 
                #'major_cities': {'query_word': ['major_cities'], 'value_type': 'list', 'value_range': []}, 
                'border_state': {'query_word': ['border','borders','bordering','surround','surrounding','neighbor','neighboring',
                                                'adjoin'], 
                          'value_type': 'string', 
                          'value_range': ['alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado', \
                                          'connecticut', 'delaware', 'florida', 'district_of_columbia', 'georgia', \
                                          'hawaii', 'idaho', 'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', \
                                          'louisiana', 'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota', \
                                          'mississippi', 'missouri', 'montana', 'nebraska', 'nevada', 'ohio', \
                                          'new_hampshire', 'new_jersey', 'new_mexico', 'new_york', 'north_carolina', \
                                          'north_dakota', 'oklahoma', 'oregon', 'pennsylvania', 'tennessee', \
                                          'rhode_island', 'south_carolina', 'south_dakota', 'texas', 'utah', 'vermont', \
                                          'virginia', 'washington', 'wisconsin', 'west_virginia', 'wyoming']}, 
                #'border': {'query_word': ['border','bordering','surround','surrounding','neighbor','neighboring'], 
                #           'value_type': 'list', 'value_range': []}, 
                #'states_in': {'query_word': ['states_in'], 'value_type': 'list', 'value_range': []}, 
                'mountain': {'query_word': ['mountain'], 'value_type': 'string', 
                             'value_range': ['mckinley', 'foraker', 'st._elias', 'bona', 'blackburn', 'kennedy', \
                                             'sanford', 'vancouver', 'south_buttress', 'churchill', 'fairweather', \
                                             'hubbard', 'bear', 'hunter', 'east_buttress', 'alverstone', 'whitney', \
                                             'browne_tower', 'elbert', 'massive', 'harvard', 'rainier', 'williamson', \
                                             'bianca', 'uncompahgre', 'la_plata', 'crestone', 'lincoln', 'grays', \
                                             'antero', 'torreys', 'castle', 'quandary', 'evans', 'longs', 'wilson', \
                                             'white', 'shavano', 'north_palisade', 'belford', 'princeton', 'yale', \
                                             'crestone_needle', 'bross', 'wrangell', 'kit_carson', 'shasta', 'sill', \
                                             'maroon', 'el_diente']}, 
                'area': {'query_word': ['area','large','largest','big','small','smallest','biggest'], 'value_type': 'int', 'value_range': []}, 
                'state': {'query_word': ['state','where','states','in','through'], 'value_type': 'string', 
                          'value_range': ['alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado', \
                                          'connecticut', 'delaware', 'florida', 'district_of_columbia', 'georgia', \
                                          'hawaii', 'idaho', 'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', \
                                          'louisiana', 'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota', \
                                          'mississippi', 'missouri', 'montana', 'nebraska', 'nevada', 'ohio', \
                                          'new_hampshire', 'new_jersey', 'new_mexico', 'new_york', 'north_carolina', \
                                          'north_dakota', 'oklahoma', 'oregon', 'pennsylvania', 'tennessee', \
                                          'rhode_island', 'south_carolina', 'south_dakota', 'texas', 'utah', 'vermont', \
                                          'virginia', 'washington', 'wisconsin', 'west_virginia', 'wyoming']}, 
                'highest_point': {'query_word': ['highest_point','peak'], 'value_type': 'string', 
                                  'value_range': ['centerville', 'cheaha_mountain', 'mount_mckinley', 'humphreys_peak', \
                                                  'magazine_mountain', 'mount_whitney', 'mount_elbert', 'mount_frissell', \
                                                  'tenleytown', 'gannett_peak', 'walton_county', 'brasstown_bald', \
                                                  'mauna_kea', 'borah_peak', 'charles_mound', 'franklin_township', \
                                                  'ocheyedan_mound', 'mount_sunflower', 'black_mountain', 'driskill_mountain', \
                                                  'mount_katahdin', 'backbone_mountain', 'mount_greylock', 'mount_curwood', \
                                                  'eagle_mountain', 'woodall_mountain', 'taum_sauk_mountain', 'granite_peak', \
                                                  'johnson_township', 'boundary_peak', 'mount_washington', 'high_point', \
                                                  'wheeler_peak', 'mount_marcy', 'mount_mitchell', 'white_butte', \
                                                  'campbell_hill', 'black_mesa', 'mount_hood', 'mount_davis', 'jerimoth_hill', \
                                                  'sassafras_mountain', 'harney_peak', 'clingmans_dome', 'guadalupe_peak', \
                                                  'kings_peak', 'mount_mansfield', 'mount_rogers', 'mount_rainier', \
                                                  'spruce_knob', 'timms_hill']}, 
                'capital': {'query_word': ['capital'], 'value_type': 'string', 
                            'value_range': ['montgomery', 'juneau', 'phoenix', 'sacramento', 'little_rock', 'denver', \
                                            'hartford', 'dover', 'washington', 'tallahassee', 'atlanta', 'honolulu', \
                                            'boise', 'springfield', 'indianapolis', 'topeka', 'des_moines', 'frankfort', \
                                            'augusta', 'baton_rouge', 'annapolis', 'boston', 'lansing', 'jackson', \
                                            'st._paul', 'helena', 'jefferson_city', 'lincoln', 'concord', 'carson_city', \
                                            'trenton', 'albany', 'santa_fe', 'raleigh', 'bismarck', 'columbus', 'salem', \
                                            'oklahoma_city', 'harrisburg', 'providence', 'columbia', 'pierre', 'nashville', \
                                            'austin', 'montpelier', 'salt_lake_city', 'richmond', 'olympia', 'charleston', \
                                            'madison', 'cheyenne']}, 
                'lowest_point': {'query_word': ['lowest_point'], 'value_type': 'string', 
                                 'value_range': ['sea_level','belle_fourche_river', 'ouachita_river', 'death_valley', 'arkansas_river', \
                                                 'long_island_sound', 'snake_river', 'verdigris_river', 'new_orleans', \
                                                 'lake_erie', 'lake_superior', 'st._francis_river', 'kootenai_river', \
                                                 'southeast_corner', 'colorado_river', 'red_bluff_reservoir', \
                                                 'red_river', 'ohio_river', 'little_river', 'delaware_river', \
                                                 'big_stone_lake', 'mississippi_river', 'gulf_of_mexico', 'beaver_dam_creek', \
                                                 'lake_champlain', 'atlantic_ocean', 'pacific_ocean', 'potomac_river', 'lake_michigan']}, 
                'major_lake': {'query_word': ['major_lake','major_lakes'], 'value_type': 'string', 
                         'value_range': ['detroit','boston','chicago','new_york','san_francisco']}, 
                'lake': {'query_word': ['lake','lakes'], 'value_type': 'string', 
                         'value_range': ['superior', 'huron', 'michigan', 'erie', 'ontario', 'iliamna', 'great_salt_lake', \
                                         'lake_of_the_woods', 'okeechobee', 'pontchartrain', 'becharof', 'red', 'champlain', \
                                         'st._clair', 'rainy', 'teshekpuk', 'salton_sea', 'naknek', 'winnebago', 'flathead', \
                                         'mille_lacs', 'tahoe']}, 
                'major_city': {'query_word': ['major_city','major_cities','big_cities'], 'value_type': 'string', 
                         'value_range': ['detroit','boston','chicago','new_york','san_francisco']}, 
                #'mountains': {'query_word': ['mountains'], 'value_type': 'list', 'value_range': []}, 
                'city': {'query_word': ['city','cities'], 'value_type': 'string', 
                         'value_range': ['birmingham', 'mobile', 'montgomery', 'huntsville', 'tuscaloosa', 'anchorage', \
                                         'phoenix', 'tucson', 'mesa', 'tempe', 'glendale', 'scottsdale', 'oakland', \
                                         'little_rock', 'fort_smith', 'north_little_rock', 'los_angeles', 'san_diego', \
                                         'san_francisco', 'san_jose', 'long_beach', 'sacramento', 'anaheim', 'fresno', \
                                         'riverside', 'santa_ana', 'stockton', 'huntington_beach', 'glendale', 'fremont', \
                                         'torrance', 'pasadena', 'garden_grove', 'san_bernardino', 'oxnard', 'east_los_angeles', \
                                         'modesto', 'sunnyvale', 'bakersfield', 'concord', 'berkeley', 'fullerton', \
                                         'inglewood', 'hayward', 'pomona', 'orange', 'ontario', 'norwalk', 'santa_monica', \
                                         'santa_clara', 'citrus_heights', 'burbank', 'downey', 'chula_vista', 'santa_rosa', \
                                         'compton', 'costa_mesa', 'carson', 'salinas', 'vallejo', 'west_covina', 'oceanside', \
                                         'el_monte', 'daly_city', 'thousand_oaks', 'san_mateo', 'simi_valley', 'richmond', \
                                         'lakewood', 'ventura', 'santa_barbara', 'el_cajon', 'westminster', 'whittier', \
                                         'alhambra', 'south_gate', 'alameda', 'buena_park', 'san_leandro', 'escondido', \
                                         'newport_beach', 'irvine', 'fairfield', 'mountain_view', 'denver', 'redondo_beach', \
                                         'scotts_valley', 'aurora', 'colorado_springs', 'lakewood', 'pueblo', 'arvada', \
                                         'boulder', 'bridgeport', 'fort_collins', 'hartford', 'waterbury', 'new_haven', \
                                         'stamford', 'norwalk', 'danbury', 'new_britain', 'west_hartford', 'greenwich', \
                                         'bristol', 'meriden', 'wilmington', 'washington', 'jacksonville', 'miami', 'tampa', \
                                         'orlando', 'st._petersburg', 'fort_lauderdale', 'hollywood', 'clearwater', 'miami_beach', \
                                         'tallahassee', 'gainesville', 'kendall', 'largo', 'west_palm_beach', 'pensacola', \
                                         'atlanta', 'columbus', 'savannah', 'macon', 'albany', 'honolulu', 'ewa', 'koolaupoko', \
                                         'boise', 'chicago', 'rockford', 'peoria', 'springfield', 'decatur', 'aurora', 'joliet', \
                                         'evanston', 'waukegan', 'elgin', 'arlington_heights', 'cicero', 'skokie', 'oak_lawn', \
                                         'champaign', 'indianapolis', 'gary', 'fort_wayne', 'evansville', 'hammond', 'south_bend', \
                                         'muncie', 'anderson', 'davenport', 'terre_haute', 'des_moines', 'cedar_rapids', 'waterloo', \
                                         'sioux_city', 'dubuque', 'wichita', 'topeka', 'louisville', 'overland_park', 'lexington', \
                                         'shreveport', 'new_orleans', 'baton_rouge', 'metairie', 'lafayette', 'kenner', 'lake_charles', \
                                         'monroe', 'portland', 'baltimore', 'dundalk', 'silver_spring', 'bethesda', 'boston', \
                                         'worcester', 'springfield', 'cambridge', 'new_bedford', 'brockton', 'lowell', 'fall_river', \
                                         'quincy', 'newton', 'lynn', 'somerville', 'framingham', 'lawrence', 'waltham', 'medford', \
                                         'detroit', 'warren', 'grand_rapids', 'flint', 'lansing', 'livonia', 'sterling_heights', \
                                         'ann_arbor', 'dearborn', 'westland', 'kalamazoo', 'taylor', 'saginaw', 'pontiac', 'southfield', \
                                         'st._clair_shores', 'clinton', 'troy', 'royal_oak', 'dearborn_heights', 'waterford', \
                                         'wyoming', 'redford', 'minneapolis', 'farmington_hills', 'duluth', 'st._paul', 'bloomington', \
                                         'rochester', 'jackson', 'springfield', 'st._louis', 'kansas_city', 'kansas_city', \
                                         'independence', 'columbia', 'st._joseph', 'billings', 'omaha', 'great_falls', 'lincoln', \
                                         'reno', 'las_vegas', 'manchester', 'nashua', 'newark', 'paterson', 'jersey_city', \
                                         'elizabeth', 'trenton', 'woodbridge', 'camden', 'clifton', 'east_orange', 'edison', \
                                         'bayonne', 'cherry_hill', 'middletown', 'irvington', 'albuquerque', 'buffalo', \
                                         'rochester', 'yonkers', 'syracuse', 'albany', 'cheektowaga', 'utica', 'schenectady', \
                                         'niagara_falls', 'new_rochelle', 'irondequoit', 'mount_vernon', 'levittown', 'charlotte', \
                                         'greensboro', 'raleigh', 'winston-salem', 'durham', 'fayetteville', 'high_point', 'fargo', \
                                         'cleveland', 'columbus', 'cincinnati', 'toledo', 'akron', 'dayton', 'youngstown', 'canton', \
                                         'parma', 'lorain', 'springfield', 'hamilton', 'lakewood', 'kettering', 'euclid', 'elyria', \
                                         'tulsa', 'oklahoma_city', 'lawton', 'norman', 'portland', 'eugene', 'salem', 'philadelphia', \
                                         'pittsburgh', 'erie', 'allentown', 'scranton', 'reading', 'upper_darby', 'bethlehem', \
                                         'abingdon', 'lower_merion', 'altoona', 'bristol_township', 'penn_hills', 'providence', \
                                         'warwick', 'cranston', 'pawtucket', 'columbia', 'charleston', 'greenville', 'north_charleston', \
                                         'memphis', 'sioux_falls', 'nashville', 'knoxville', 'chattanooga', 'houston', 'dallas', \
                                         'austin', 'san_antonio', 'el_paso', 'fort_worth', 'lubbock', 'corpus_christi', 'arlington', \
                                         'amarillo', 'garland', 'beaumont', 'pasadena', 'irving', 'waco', 'abilene', 'laredo', \
                                         'wichita_falls', 'odessa', 'brownsville', 'richardson', 'san_angelo', 'plano', 'midland', \
                                         'grand_prairie', 'tyler', 'mesquite', 'mcallen', 'longview', 'provo', 'port_arthur', \
                                         'salt_lake_city', 'ogden', 'west_valley', 'norfolk', 'richmond', 'virginia_beach', 'arlington', \
                                         'hampton', 'newport_news', 'chesapeake', 'portsmouth', 'alexandria', 'roanoke', 'lynchburg', \
                                         'seattle', 'spokane', 'tacoma', 'bellevue', 'charleston', 'huntington', 'milwaukee', \
                                         'madison', 'racine', 'green_bay', 'kenosha', 'appleton', 'west_allis', 'casper','washington_dc',\
                                         'new_york_city']}, 
                'country': {'query_word': [], 'value_type': 'string', 
                            'value_range': ['us', 'united_states', 'usa','america']}, 
                #'states_through': {'query_word': ['states_through'], 'value_type': 'list', 'value_range': []}, 
                #'lakes': {'query_word': ['lakes'], 'value_type': 'list', 'value_range': []}, 
                #'rivers': {'query_word': ['rivers'], 'value_type': 'list', 'value_range': []}, 
                'population': {'query_word': ['population','populous','people','citizens','residents','inhabitants'], 
                                'value_type': 'int', 'value_range': []}, 
                'length': {'query_word': ['length','longest','long','longer','shortest','short'], 'value_type': 'int', 'value_range': []}, 
                'major_river': {'query_word': ['major_river','major_rivers'], 'value_type': 'string', 
                          'value_range': ['mississippi', 'missouri', 'arkansas', 'rio_grande', 'columbia', 'hudson', 'colorado']},
                'river': {'query_word': ['river','rivers','runs','run','go','goes','traverse','traverses'], 
                          'value_type': 'string', 
                          'value_range': ['mississippi', 'missouri', 'colorado', 'ohio', 'red', 'arkansas', 'canadian', 'connecticut', \
                                          'delaware', 'snake', 'little_missouri', 'chattahoochee', 'cimarron', 'green', 'potomac', \
                                          'north_platte', 'republican', 'tennessee', 'rio_grande', 'san_juan', 'wabash', 'yellowstone', \
                                          'allegheny', 'bighorn', 'cheyenne', 'columbia', 'clark_fork', 'cumberland', 'dakota', 'gila', \
                                          'hudson', 'neosho', 'niobrara', 'ouachita', 'pearl', 'pecos', 'powder', 'roanoke', 'rock', \
                                          'tombigbee', 'smoky_hill', 'south_platte', 'st._francis', 'washita', 'white', 'wateree_catawba']}
               }
