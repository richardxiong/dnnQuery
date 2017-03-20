### Developed by: Hongyu, Kaifeng and Yinglan
### the grammar rules are extended from the synthetic dataset of Neural Enquirer

conventions:
1. all punctuations are deleted in both questions and logical forms, meaning no ',' '.' '?'
2. '#' is replaced with 'number'
3. 'arg max'=>'argmax', 'arg min'=>'argmin'
4. '=' => 'equal'
5. '<' => 'less', '>' => 'greater

[Please read the replaceDic in dataset.py]
I. DIRECT QUERIES: 6 possible logical forms (and their inferences):
===========================================
length = 3
utterance_superlative_query = 'arg {max_min} {arg1} {arg2}'

examples:
1. how many medals are in the game with the most medals
argmax number_medals number_medals

2. how wealthy is the host country of the game hosted by the largest country
argmax country_gdp country_size

===========================================
length = 6
utterance_pattern_single_select = 'where {comp_field} = {comp_val}  ' \
                                  'select {select_field}'

examples:
1. how long is the game with 30 medals
where number_medals equal 30 select number_duration

2. how many people participated in the game whose host country population is 300
where country_population equal 300 select number_participants

===========================================
length = 7
utterance_pattern_multi_field = 'where {comp_field} {comp} {comp_val} ' \
                                ' arg {max_min} {arg1} {arg2}'

examples:
1. when was the game hosted watched by the most people whose number of medals is smaller than 420
where number_medals less 420 argmax year number_audience

2. when was the game with the largest host country population whose number of audience is larger than 40
where number_audience greater 40 argmax year country_population

===========================================
length = 10 (new)
utterance_pattern_multi_field = 'where {comp_field1} {comp} {comp_val1} ' \
                                'where {comp_field2} {comp} {comp_val2}  ' \
                                  'select {select_field}'

examples:
1. how many people participated in the game whose host country population is 300 and whose number of medals is smaller than 420
where number_medals less 420 where country_population equal 300 select number_participants

2. What's the rank of nation who has 3 gold medals and 4 silver medals?
where Gold equal 3 where Silver equal 4 select Nation

===========================================
length = 11 (new)
utterance_pattern_multi_field = 'where {comp_field1} {comp} {comp_val1} ' \
                                'where {comp_field2} {comp} {comp_val2}  ' \
                                  ' arg {max_min} {arg1} {arg2}'

examples:
1. which nation with 2 gold and 0 silver has the most medals in total
where Gold equal 2 where Silver equal 0 argmax(Nation, Total)


===========================================
length = 15
utterance_nested_query_4field = 'where {query1_comp_field} {query1_comp} {query1_comp_val} ' \
                                          'select {query1_project_field} as A ' \
                                          'where {query1_project_field} {query2_comp} A ' \
                                          'arg {max_min} {arg1} {arg2}'

examples:
1. how long is the game with the most medals whose host country is less wealthy than the game in seattle
where host_city equal Seattle select country_gdp as A where country_gdp less A argmax number_duration number_medals

2. how many people watched the latest game that lasts for more days than the game in 2044
where year equal 2044 select number_duration as A where number_duration greater A argmax number_audience year

3. which team was the next winner after ballyroan abbey
where Team equal Ballyroan_abbey select Years as A where Years greater A argmin Team Year

===========================================

**************************************************************************************
II. QUERIES BASED ON CALCULATION
sum total
average
subtract

===========================================
sum: sum([field], [A, B, C...])

length = 1
utterance_sum_basic = 'sum'

length = 2
utterance_sum_basic2 = 'sum {query_field}'

e.g. what is the total number of gold medals earned
sum Gold

length = 5
utterance_sum_dependent = 'where {comp_field1} {comp} {comp_val} ' \
					'sum'

e.g. what is the total number of nations that did not win gold
where Gold equal 0 sum

length = 6
utterance_sum_dependent = 'where {comp_field1} {comp} {comp_val} ' \
					'sum {comp_field2}'

length = 19
utterance_sum_multi = 'where {query1_comp_field} {query1_comp} {query1_comp_val} ' \
                                          'select {query1_project_field} as A ' \
                      'where {query1_comp_field} {query2_comp} {query2_comp_val} ' \
                                          'select {query2_project_field} as B ' \
                      ...
                      sum A B ...

e.g. how many winning golfers does england and wales combined have in the masters
where Country equal England select Masters as A where Country equal Wales select Masters as B sum A B

============================================
mean: mean([field], [A, B, C...])

length = 2
utterance_mean_basic = 'mean {query_field}'

length = 6
utterance_mean_dependent = 'where {comp_field1} {comp} {comp_val} ' \
						'mean {comp_field2}'

e.g. what is the mean number of total appearances of scotland
where Nation equal Scotland mean Total_Apps

length = 19
utterance_mean_multi = 'where {query1_comp_field} {query1_comp} {query1_comp_val} ' \
                                          'select {query1_project_field} as A ' \
                      'where {query1_comp_field} {query2_comp} {query2_comp_val} ' \
                                          'select {query2_project_field} as B ' \
                      ...
                      mean A B ...

e.g. what is the average of silver medals earned by korea and japan
where Nation equal Korea select Silver as A where Nation equal Japan select Silver as B mean A B

============================================
diff: diff(A, B)

length = 13
utterance_diff_1 = 	' arg {max_min} {arg1} {arg2} as A ' \
                      ' arg {max_min} {arg1} {arg2} as B ' \
                      diff A B

e.g.
1 what is the difference between the nation with the most medals and the nation with the least amount of medals
argmax Total Total as A argmin Total Total as B diff A B
2 largest medal differential between countries
argmax Total Total as A argmin Total Total as B diff A B

length = 19
utterance_diff_2 = 'where {query1_comp_field} {query1_comp} {query1_comp_val} ' \
                                          'select {query1_project_field} as A ' \
                      'where {query1_comp_field} {query2_comp} {query2_comp_val} ' \
                                          'select {query2_project_field} as B ' \
                      diff A B

e.g. what is the difference in total number of medals between india and nepal
where Nation equal India select Total as A where Nation equal Nepal select Total as B diff A B

**************************************************************************************
III. COUPLING BETWEEN MULTIPLE FIELDS
e.g. which of the girls had the least amount in archery (girls are fields)

The query of a given logical form is in logicalParser
