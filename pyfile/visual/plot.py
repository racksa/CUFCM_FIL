import os
import visual
import sys

# initialise the visuliser
visualiser = visual.VISUAL()

# lists of available options
list_of_single_options = ['plot', 'plot_eco', 'plot_fil', 'phase', 'wavenumber', 'phase_plane', 'order_parameter', 'eckert', 'ciliate', 'ciliate_eco', 'ciliate_speed', 'ciliate_speed_eco', 'ciliate_traj', 
                   'timing', 'ciliate_forcing', 'ciliate_dissipation', 'footpath', 'phase_diff',
                   'ciliate_svd', 'ciliate_dmd', 'kymograph', 'copy_phases',
                   'periodic_solution', 'find_periodicity', 'spherical_contour', 'flow_field_2D', 'flow_field_kymograph', 'flow_field_FFCM','flow_field_polar','flow_field_FFCM_series', 'ciliate_2D']
# list_of_multi_options = ['multi_phase', 'multi_ciliate', 'multi_ciliate_traj',
#                          'multi_ciliate_speed', 'multi_timing', 'multi_ciliate_svd',
#                          'multi_check_overlap', 'multi_ciliate_dissipation',
#                          'multi_order_parameter', 'multi_ciliate_dissipation_generate',
#                          'multi_ciliate_dissipation_plots', 'multi_output_phase',
#                          'multi_kymograph', 'multi_copy_lastline_phases']
# list_of_summary_options = ['summary_ciliate_speed', 'summary_timing',
#                            'summary_ciliate_dissipation', 'summary_check_overlap',
#                            'summary_ciliate_density', 'summary_ciliate_k', 'summary_ciliate_resolution']

# list_of_special_options = ['ishikawa', 'view_solution', 'mod_state']
# list_of_all_options = list_of_single_options\
#                     + list_of_multi_options\
#                     + list_of_summary_options\
#                     + list_of_special_options

attributes = dir(visualiser)
methods = [attr for attr in attributes if callable(getattr(visualiser, attr)) and not attr.startswith('__') and not attr.endswith('__')]
# methods = list_of_all_options

# execute the plotting function
if(sys.argv[1] in methods):
    visualiser.read_rules()
    if('interpolate' in sys.argv):
        visualiser.interpolate = True
    if('angle' in sys.argv):
        visualiser.angle = True
    if('check_overlap' in sys.argv):
        visualiser.check_overlap = True

    if(sys.argv[1] in list_of_single_options):
        if(len(sys.argv) > 2):
            if(sys.argv[2].isdigit()):
                visualiser.index = int(sys.argv[2])
        if('video' in sys.argv):
            visualiser.video = True

        if('plane' in sys.argv):
            visualiser.planar = True
            visualiser.big_sphere = False
            visualiser.show_poles = False

        if('blob' in sys.argv):
            visualiser.big_sphere = False
            visualiser.noblob = False
        
    if hasattr(visualiser, sys.argv[1]):
        method_to_call = getattr(visualiser, sys.argv[1])
        if callable(method_to_call):
            method_to_call()
        else:
            print(f"'{sys.argv[1]}' is not a callable method.")

if(sys.argv[1] == 'special'):
    visualiser.multi_ciliate_special_func()

if(sys.argv[1] == 'special2'):
    visualiser.multi_ciliate_special_func2()









#