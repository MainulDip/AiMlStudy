const featureFunctions = {}

featureFunctions.getPathCount = (paths) => {
    return paths.length
}

featureFunctions.getPointCount = (paths) => {
    const points = paths.flat() // converting multidimensional array into single dimensional array
    return points.length
}

if (typeof module !== 'undefined') {
    module.exports = featureFunctions
}