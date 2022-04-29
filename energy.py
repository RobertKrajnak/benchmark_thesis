from pyJoules.energy_meter import measure_energy

@measure_energy
def foo():
    print("ahoj")

foo()