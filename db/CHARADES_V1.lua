require 'paths'
require 'sys'
local ffi = require 'ffi'

local function strTableToTensor( strTable )
	local maxStrLen = 0
	local numStr = #strTable
	for _, path in pairs( strTable ) do
		if maxStrLen < path:len(  ) then maxStrLen = path:len(  ) end
	end
	maxStrLen = maxStrLen + 1
	local charTensor = torch.CharTensor( numStr, maxStrLen ):fill( 0 )
	local pt = charTensor:data(  )
	for _, path in pairs( strTable ) do
		ffi.copy( pt, path )
		pt = pt + maxStrLen
	end
	-- Manual clean up. Fuck lua gargabe collection!!!
	for i = 1, #strTable do strTable[ i ] = nil end strTable = nil
	collectgarbage(  )
	return charTensor
end

function genDb( setName )
	local rootDir = gpath.db.charades_v1
	local frameDir = paths.concat( rootDir, 'data' )
	-- Make cid2name.
	local cid2name = {  }
	local cid = 0
	for line in io.lines( paths.concat( rootDir, 'db_cid2name.txt' ) ) do
		cid = cid + 1
		cid2name[ cid ] = line
	end
	cid2name = strTableToTensor( cid2name )
	-- Make vid2path.
	local vid2path = {  }
	local vid = 0
	for line in io.lines( paths.concat( rootDir, 'db_' .. setName .. '_vid2name.txt' ) ) do
		vid = vid + 1
		vid2path[ vid ] = paths.concat( frameDir, line )
	end
	vid2path = strTableToTensor( vid2path )
	-- Make vid2numf.
	local vid2numf = torch.IntTensor( vid2path:size( 1 ) ):fill( 0 )
	local vid = 0
	for line in io.lines( paths.concat( rootDir, 'db_' .. setName .. '_vid2numf.txt' ) ) do
		vid = vid + 1
		vid2numf[ vid ] = tonumber( line )
	end
	assert( vid == vid2path:size( 1 ) )
	-- Make aid2vid.
	local na = 0
	for line in io.lines( paths.concat( rootDir, 'db_' .. setName .. '_aid2vid.txt' ) ) do
		na = na + 1
	end
	local aid2vid = torch.IntTensor( na ):fill( 0 )
	local aid = 0
	for line in io.lines( paths.concat( rootDir, 'db_' .. setName .. '_aid2vid.txt' ) ) do
		aid = aid + 1
		aid2vid[ aid ] = tonumber( line )
	end
	-- Make aid2cid.
	local aid2cid = torch.IntTensor( aid2vid:numel(  ) ):fill( 0 )
	local aid = 0
	for line in io.lines( paths.concat( rootDir, 'db_' .. setName .. '_aid2cid.txt' ) ) do
		aid = aid + 1
		aid2cid[ aid ] = tonumber( line )
	end
	assert( aid == aid2vid:numel(  ) )
	-- Make aid2loc.
	local aid2loc = torch.IntTensor( aid2vid:numel(  ), 2 ):fill( 0 )
	local vid2bias = torch.IntTensor( { 0 } ):cat( vid2numf:cumsum(  )[ { { 1, -2 } } ] )
	local aid = 0
	for line in io.lines( paths.concat( rootDir, 'db_' .. setName .. '_aid2loc.txt' ) ) do
		aid = aid + 1
		local vid = aid2vid[ aid ]
		local bias = vid2bias[ vid ]
		local loc = line:split( ' ' )
		local ts = tonumber( loc[ 1 ] )
		local te = tonumber( loc[ 2 ] )
		aid2loc[ aid ][ 1 ] = ts + bias
		aid2loc[ aid ][ 2 ] = te + bias
		assert( ts < te )
		assert( te <= vid2numf[ vid ] )
	end
	assert( aid == aid2vid:numel(  ) )
	-- data verification.
	assert( cid2name:size( 1 ) >= aid2cid:max(  ) )
	assert( vid2path:size( 1 ) >= aid2vid:max(  ) )
	assert( ( aid2loc[ { {  }, { 2 } } ] - aid2loc[ { {  }, { 1 } } ] ):gt( 0 ):sum(  ) == aid2loc:size( 1 ) )
	assert( aid2loc[ -1 ][ 1 ] > vid2numf[ { { 1, -2 } } ]:sum(  ) )
	assert( aid2loc[ -1 ][ 2 ] <= vid2numf:sum(  ) )
	-- done.
	collectgarbage(  )
	return vid2path, vid2numf, cid2name, aid2cid, aid2loc, aid2vid, '%04d.jpg'
end
