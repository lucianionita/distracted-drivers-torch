-- Aux function to generate a string form a table of drivers
-------------------------------------------------------------
function string_drivers(drivers)
    s = ""
    for k,v in pairs(drivers) do
        s = s .. " " .. v
    end
    s = s .. " "
    return s
end




