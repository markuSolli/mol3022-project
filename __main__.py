import dearpygui.dearpygui as dpg

dpg.create_context()
dpg.create_viewport(width=700, height=300, title='Protein secondary structure prediction tool')
dpg.setup_dearpygui()

def print_callback(_, app_data, user_data):
    print(dpg.get_value('sequence_input_text'))

with dpg.window() as primary_window:
    dpg.add_text('Sequence')
    dpg.add_input_text(tag='sequence_input_text')
    dpg.add_button(label='Print sequence', callback=print_callback)

dpg.set_primary_window(primary_window, True)
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()