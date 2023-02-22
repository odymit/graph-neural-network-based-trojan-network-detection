import pstats as ps

p = ps.Stats('time_cost_0.prof')
p.sort_stats('time').print_stats()
