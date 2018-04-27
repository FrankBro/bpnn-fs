module Bpnn

open System

module Random =
    let doubleInRange min max (rand: Random) =
        rand.NextDouble() * (max - min) + min

module String =
    let join (sep: string) (xs: #seq<string>) = String.Join(sep, xs)

type Network = {
    InputActivations: float []
    HiddenActivations: float []
    OutputActivations: float []
    InputWeights: float [] []
    OutputWeights: float [] []
    InputChanges: float [] []
    OutputChanges: float [] []
}

let sigmoid x = tanh x
let dsigmoid x = 1. - x ** 2.

let initNetwork ni nh no (random: Random) =
    let ni = ni + 1
    {
        InputActivations = Array.init ni (fun _ -> 1.)
        HiddenActivations = Array.init nh (fun _ -> 1.)
        OutputActivations = Array.init no (fun _ -> 1.)
        InputWeights =
            Array.init ni (fun _ ->
                Array.init nh (fun _ ->
                    Random.doubleInRange -0.2 0.2 random
                )
            )
        OutputWeights =
            Array.init nh (fun _ ->
                Array.init no (fun _ ->
                    Random.doubleInRange -2.0 2.0 random
                )
            )
        InputChanges =
            Array.init ni (fun _ ->
                Array.init nh (fun _ ->
                    0.
                )
            )
        OutputChanges =
            Array.init nh (fun _ ->
                Array.init no (fun _ ->
                    0.
                )
            )
    }

let update (inputs: float []) network =
    if Array.length inputs <> Array.length network.InputActivations - 1 then
        failwith "Wrong number of inputs"
    let inputActivations =
        Array.last network.InputActivations
        |> Array.singleton
        |> Array.append inputs 
    let hiddenActivations =
        Array.init (Array.length network.HiddenActivations) (fun j ->
            network.InputActivations
            |> Array.mapi (fun i activation ->
                activation * network.InputWeights.[i].[j]
            )
            |> Array.sum
        )
    let outputActivations =
        Array.init (Array.length network.OutputActivations) (fun k ->
            network.HiddenActivations
            |> Array.mapi (fun j activation ->
                activation * network.OutputWeights.[j].[k]
            )
            |> Array.sum
        )
    { network with
        InputActivations = inputActivations
        HiddenActivations = hiddenActivations
        OutputActivations = outputActivations
    }

let backPropagate targets n m network =
    if Array.length targets <> Array.length network.OutputActivations then
        failwith "Wrong number of targets"
    let outputDeltas =
        network.OutputActivations
        |> Array.mapi (fun k activation ->
            let error = targets.[k] - activation
            dsigmoid activation * error
        )
    let hiddenDeltas =
        network.HiddenActivations
        |> Array.mapi (fun j activation ->
            let error =
                outputDeltas
                |> Array.mapi (fun k delta ->
                    delta * network.OutputWeights.[j].[k]
                )
                |> Array.sum
            dsigmoid activation * error
        )
    let outputChanges =
        network.HiddenActivations
        |> Array.map (fun activation ->
            outputDeltas
            |> Array.map (fun delta ->
                delta * activation
            )
        )
    let outputWeights =
        outputChanges
        |> Array.mapi (fun j outputChange ->
            outputChange
            |> Array.mapi (fun k change ->
                network.OutputWeights.[j].[k] + n * change + m * network.OutputChanges.[j].[k]
            )
        )
    let inputChanges =
        network.InputActivations
        |> Array.map (fun activation ->
            hiddenDeltas
            |> Array.map (fun delta ->
                delta * activation
            )
        )
    let inputWeights =
        inputChanges
        |> Array.mapi (fun i inputChange ->
            inputChange
            |> Array.mapi (fun j change ->
                network.InputWeights.[i].[j] + n * change + m * network.InputChanges.[i].[j]
            )
        )
    let network =
        { network with
            OutputWeights = outputWeights
            OutputChanges = outputChanges
            InputWeights = inputWeights
            InputChanges = inputChanges
        }
    let error =
        targets
        |> Array.mapi (fun k target ->
            0.5 * (target - network.OutputActivations.[k]) ** 2.
        )
        |> Array.sum
    network, error

let test (patterns: (float [] * float []) []) network =
    for (pattern, _) in patterns do
        let patternString =
            pattern
            |> Array.map string
            |> String.join ", "
        let resultString =
            (update pattern network).OutputActivations
            |> Array.map (sprintf "%.2f")
            |> String.join ", "
        printfn "[%s] -> [%s]" patternString resultString

let train patterns iter n m network =
    (network, [|1..iter|])
    ||> Array.fold (fun network _ ->
        (network, patterns)
        ||> Array.fold (fun network pattern ->
            let inputs = fst pattern
            let targets = snd pattern
            let network = update inputs network
            backPropagate targets n m network
            |> fst
        )
    )

let benchmark () =
    // XOR
    let patterns = [|
        [|-1.; -1.|], [|-1.|]
        [|-1.; 1. |], [|1. |]
        [|1.; -1. |], [|1. |]
        [|1.; 1.  |], [|-1.|]
    |]
    let random = Random()
    let network = initNetwork 2 3 1 random
    let network = train patterns 1000 0.5 0.1 network
    test patterns network

[<EntryPoint>]
let main argv =
    benchmark ()
    0 // return an integer exit code
