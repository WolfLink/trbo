from bqskit.ir import Circuit

from bqskit.ir.gates.constant import HGate, ZGate, XGate, TGate, TdgGate, SGate, SdgGate, CNOTGate
from bqskit.ir.gates.parameterized import RZGate
from bqskit.ir.gates.composed import ControlledGate


circuit = 0


gatemap = {
        "H" : HGate(),
        "Z" : ZGate(),
        "not" : XGate(),
        "T" : [TGate(), TdgGate()],
        "S" : [SGate(), SdgGate()]
        }

def parse_quipper_file(filename):
    qubit_names = []
    circuit = None
    with open(filename, "r") as f:
        for line in f.readlines():
            try:
                if line.startswith("Inputs"):
                    segments = line.split(":Qbit")
                    qubit_names.append(segments[0].split('Inputs: ')[1])
                    for segment in segments[1:-1]:
                        qubit_names.append(segment.split(", ")[1])
                    circuit = Circuit(len(qubit_names))
                    # for now I am assuming qubits are named in order starting from 0 and none are unused

                elif line.startswith("QGate"):
                    segments = line.split(" ")
                    gatebase = segments[0]
                    dagger = "]*(" in gatebase
                    if not dagger:
                        qubit_index = int(gatebase.split('QGate["')[1].split('"](')[1].split(')')[0])
                        gatename = gatebase.split('QGate["')[1].split('"](')[0]
                    else:
                        qubit_index = int(gatebase.split('QGate["')[1].split('"]*(')[1].split(')')[0])
                        gatename = gatebase.split('QGate["')[1].split('"]*(')[0]

                    controls = []
                    for segment in segments:
                        if "controls=[" in segment:
                            for control in segment.split('controls=[')[1].split(','):
                                control = int(control.split(']')[0])
                                if control < 0:
                                    print("I haven't figured out how to handle negative controls yet")
                                    raise SyntaxError()
                                else:
                                    controls.append(control)

                    if gatename in ["T", "S"]:
                        gate = gatemap[gatename][1 if dagger else 0]
                    else:
                        gate = gatemap[gatename]

                    if len(controls) == 1 and gatename == "not":
                        circuit.append_gate(CNOTGate(), controls + [qubit_index])
                    elif len(controls) > 0:
                        circuit.append_gate(ControlledGate(gate, len(controls)), controls + [qubit_index])
                    else:
                        circuit.append_gate(gate, [qubit_index])

                elif line.startswith('QRot["exp(-i%Z)",'):
                    angle = float(line.split('QRot["exp(-i%Z)",')[1].split("](")[0])
                    qubit_index = int(line.split('QRot["exp(-i%Z)",')[1].split("](")[1].split(")")[0])
                    circuit.append_gate(RZGate(), [qubit_index], [angle])

                elif line.startswith("Outputs"):
                    continue
                elif line.startswith("Comment"):
                    continue
                elif line.startswith("QInit"):
                    continue # maybe we could take advantage of this eventually but I don't think its relevant for the current project
                elif line in ["\n", "\r\n", " \n", " \r\n"]:
                    continue
                else:
                    raise SyntaxError()
            except:
                print("There was an error parsing the line: ")
                print(line)
                raise

    return circuit

