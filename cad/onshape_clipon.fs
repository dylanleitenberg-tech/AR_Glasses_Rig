// ⚠️ DEPRECATED (2026-07-06 audit): positions/geometry here are STALE — several design
// revisions behind cad/xreal_one_mount.scad. The canonical OnShape route is STL IMPORT
// of the verified export (openscad -D 'part="carrier"' -o carrier.stl xreal_one_mount.scad).
// Kept only as a FeatureScript reference.
FeatureScript 2278;
import(path : "onshape/std/geometry.fs", version : "2278.0");
import(path : "onshape/std/boolean.fs", version : "2278.0");   // BooleanOperationType + opBoolean

// =====================================================================
//  AR eye-tracking CLIP-ON carrier — parametric, from the VALIDATED rig geometry.
//  Frame: x = right, y = forward (world), z = up; origin = brow-rail centre.
//  USE: new Feature Studio -> paste BELOW its auto-generated first two lines
//       (keep Onshape's version + import) -> Commit -> Insert "AR clip-on carrier".
// =====================================================================

// parameter bounds (mm: [min, default, max])
export const IPD_BOUNDS  = { (millimeter) : [50, 67, 80] } as LengthBoundSpec;
export const DROP_BOUNDS = { (millimeter) : [10, 25.3, 40] } as LengthBoundSpec;
export const BOOM_BOUNDS = { (millimeter) : [3, 7, 12] } as LengthBoundSpec;

function uvec(v is Vector) returns Vector
{
    return v / norm(v);
}

// sim (x=right, y=up, z=fwd) -> CAD (x=right, y=fwd, z=up), origin at brow rail
function toCad(s is array, drop is number) returns Vector
{
    return vector(s[0], s[2], s[1] - drop) * millimeter;
}

// one camera holder: lens-shroud cylinder + backing disc, optical axis C -> target
function camHolder(context is Context, id is Id, C is Vector, target is Vector, board is ValueWithUnits)
{
    const ax = uvec(target - C);
    const back = 12 * millimeter;
    const front = 7 * millimeter;
    fCylinder(context, id + "shroud", {
            "topCenter" : C + front * ax,
            "bottomCenter" : C - (back + 2 * millimeter) * ax,
            "radius" : 11 * millimeter });
    fCylinder(context, id + "disc", {
            "topCenter" : C - back * ax,
            "bottomCenter" : C - (back + 2.6 * millimeter) * ax,
            "radius" : board / 2 * 0.95 });
}

function boom(context is Context, id is Id, a is Vector, b is Vector, dia is ValueWithUnits)
{
    fCylinder(context, id, { "topCenter" : a, "bottomCenter" : b, "radius" : dia / 2 });
}

annotation { "Feature Type Name" : "AR clip-on carrier" }
export const arClipon = defineFeature(function(context is Context, id is Id, definition is map)
    precondition
    {
        annotation { "Name" : "Pupil IPD" }
        isLength(definition.ipd, IPD_BOUNDS);
        annotation { "Name" : "Optic drop" }
        isLength(definition.opticDrop, DROP_BOUNDS);
        annotation { "Name" : "Boom diameter" }
        isLength(definition.boomD, BOOM_BOUNDS);
        annotation { "Name" : "Build stereo (8-cam)" }
        definition.stereo is boolean;
    }
    {
        const ipd = definition.ipd / millimeter;
        const drop = definition.opticDrop / millimeter;
        const h = ipd / 2;

        const canthR = [44.5, 3.82, -13.39];
        const canthL = [-44.5, 3.82, -13.39];
        const corR = [h, 0, -28.5];
        const corL = [-h, 0, -28.5];

        var cams = [
            { "C" : [h, 30, 5],        "T" : [h, 30, 105],  "b" : 38 },
            { "C" : [-h, 30, 5],       "T" : [-h, 30, 105], "b" : 38 },
            { "C" : [h + 36, -2, -6],  "T" : canthR,        "b" : 36 },
            { "C" : [-(h + 36), -2, -6], "T" : canthL,      "b" : 36 },
            { "C" : [h - 8, -33, 6],   "T" : corR,          "b" : 36 },
            { "C" : [-(h - 8), -33, 6], "T" : corL,         "b" : 36 }
        ];
        if (definition.stereo)
        {
            cams = append(cams, { "C" : [h + 36, -16, -10],   "T" : canthR, "b" : 36 });
            cams = append(cams, { "C" : [-(h + 36), -16, -10], "T" : canthL, "b" : 36 });
        }

        // wider brow bar (12 mm deep x 8 mm tall) for support
        const railHalf = (h + 44) * millimeter;
        fCuboid(context, id + "rail", {
                "corner1" : vector(-railHalf, -6 * millimeter, -4 * millimeter),
                "corner2" : vector(railHalf, 6 * millimeter, 4 * millimeter) });

        // holders + booms (L-routing for side/under-eye cams)
        for (var i = 0; i < size(cams); i += 1)
        {
            const C = toCad(cams[i].C, drop);
            const T = toCad(cams[i].T, drop);
            camHolder(context, id + ("h" ~ toString(i)), C, T, cams[i].b * millimeter);
            const back = C - uvec(T - C) * 12 * millimeter;
            const railPt = vector(C[0], 0 * millimeter, 0 * millimeter);
            if (C[2] < -15 * millimeter)
            {
                const dropPt = vector(C[0], 0 * millimeter, C[2]);
                boom(context, id + ("bd" ~ toString(i)), railPt, dropPt, definition.boomD);
                boom(context, id + ("br" ~ toString(i)), dropPt, back, definition.boomD);
            }
            else
            {
                boom(context, id + ("b" ~ toString(i)), railPt, back, definition.boomD);
            }
        }

        // brow clamp stubs
        for (var sgn in [-1, 1])
        {
            const cx = sgn * 38 * millimeter;
            fCuboid(context, id + ("clamp" ~ toString(sgn)), {
                    "corner1" : vector(cx - 7 * millimeter, -10 * millimeter, -9 * millimeter),
                    "corner2" : vector(cx + 7 * millimeter, 10 * millimeter, 9 * millimeter) });
        }

        // IMU pocket block
        fCuboid(context, id + "imu", {
                "corner1" : vector(-12 * millimeter, 3 * millimeter, -2 * millimeter),
                "corner2" : vector(12 * millimeter, 25 * millimeter, 4 * millimeter) });

        // merge everything into one solid
        opBoolean(context, id + "unionAll", {
                "tools" : qCreatedBy(id, EntityType.BODY),
                "operationType" : BooleanOperationType.UNION });

        // clear each camera's optical path LAST so nothing blocks a cam
        var bores = [];
        for (var i = 0; i < size(cams); i += 1)
        {
            const C = toCad(cams[i].C, drop);
            const ax = uvec(toCad(cams[i].T, drop) - C);
            fCylinder(context, id + ("bore" ~ toString(i)), {
                    "topCenter" : C + 9 * millimeter * ax,
                    "bottomCenter" : C - 16 * millimeter * ax,
                    "radius" : 9 * millimeter });
            bores = append(bores, qCreatedBy(id + ("bore" ~ toString(i)), EntityType.BODY));
        }
        opBoolean(context, id + "clearCams", {
                "tools" : qUnion(bores),
                "targets" : qCreatedBy(id + "unionAll", EntityType.BODY),
                "operationType" : BooleanOperationType.SUBTRACTION });
    });
