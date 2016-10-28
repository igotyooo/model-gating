---------------------------------------------------------
-- This file was customized from Soumith's "dataset.lua".
---------------------------------------------------------

require 'torch'
require 'image'
local ffi = require 'ffi'
paths.dofile( string.format( './db/%s.lua', opt.data ) )
torch.setdefaulttensortype( 'torch.FloatTensor' )

local dataset = torch.class( 'dataLoader' )
function dataset:__init( gendb, setName, loadSize, sampleSize )
	self.setName = setName
	self.loadSize = loadSize
	self.sampleSize = sampleSize
	self.sampleHookTrain = self.defaultSampleHook
	self.sampleHookTest = self.defaultSampleHook
	self.vid2path, self.vid2numim, self.cid2name, self.aid2cid, self.aid2loc, self.aid2vid, self.frameFormat = genDb( setName )
	-- Make vid2ffid.
	self.vid2ffid = torch.IntTensor( { 0 } ):cat( self.vid2numim:cumsum(  )[ { { 1, -2 } } ] ) + 1
	-- Make cid2numa.
	local numCls = self.cid2name:size( 1 )
	local cid2numa = torch.IntTensor( numCls )
	for cid = 1, numCls do
		cid2numa[ cid ] = self.aid2cid:eq( cid ):sum(  )
	end
	self.cid2numa = cid2numa
	-- Make cid2aids.
	local cid2aids = torch.IntTensor( numCls, cid2numa:max(  ) ):fill( 0 )
	local aids = torch.range( 1, self.aid2vid:numel(  ) ):type( 'torch.IntTensor' )
	for cid = 1, numCls do
		cid2aids[ { cid, { 1, cid2numa[ cid ] } } ]:copy( aids[ self.aid2cid:eq( cid ) ] )
	end
	assert( cid2aids:max(  ) == self.aid2cid:numel(  ) )
	self.cid2aids = cid2aids
	-- Make fid2cids.
	local fid2cids_ = {  }
	for aid = 1, self.aid2loc:size( 1 ) do
		local fids = self.aid2loc[ aid ][ 1 ]
		local fide = self.aid2loc[ aid ][ 2 ]
		local cid = self.aid2cid[ aid ]
		for fid = fids, fide do
			if fid2cids_[ fid ] == nil then fid2cids_[ fid ] = {  }	end
			table.insert( fid2cids_[ fid ], cid )
		end
	end
	local numMaxAct = 0
	for fid, cids in pairs( fid2cids_ ) do
		if #cids > numMaxAct then numMaxAct = #cids end
	end
	local fid2cids = torch.IntTensor( self.vid2numim:sum(  ), numMaxAct ):fill( 0 )
	for fid, cids in pairs( fid2cids_ ) do
		fid2cids[ fid ][ { { 1, #cids } } ]:copy( torch.IntTensor( cids ) )
	end
	self.fid2cids = fid2cids
	collectgarbage(  )
end
-- Converts a table of samples and labels to a clean tensor.
local function tableToOutput( self, dataTable, labelTable )
	local data, labels
	local quantity = #labelTable
	local numDim = self.cid2name:size( 1 )
	assert( dataTable[ 1 ]:dim(  ) == 3 )
	assert( #dataTable == #labelTable )
	data = torch.Tensor( quantity, 3, self.sampleSize, self.sampleSize )
	labels = torch.IntTensor( quantity, numDim ):fill( 0 )
	for i = 1, quantity do
		data[ i ]:copy( dataTable[ i ] )
		local cids = labelTable[ i ][ labelTable[ i ]:gt( 0 ) ]
		labels[ i ][ { { 1, cids:size( 1 ) } } ]:copy( cids )
	end
	return data, labels
end
function dataset:sampleActions( quantity, seqLength )
	assert( quantity > 0 )
	assert( ( quantity % seqLength ) == 0 )
	local dataTable = {  }
	local labelTable = {  }
	local numActionToSample = quantity / seqLength
	for a = 1, numActionToSample do
		local cid = torch.random( 1, self.cid2name:size( 1 ) )
		local aid = self.cid2aids[ cid ][ torch.random( 1, self.cid2numa[ cid ] ) ]
		local vid = self.aid2vid[ aid ]
		local vpath = ffi.string( torch.data( self.vid2path[ vid ] ) )
		local fids = self.aid2loc[ aid ][ 1 ]
		local fide = self.aid2loc[ aid ][ 2 ]
		local ffid = self.vid2ffid[ vid ]
		local fids = torch.random( fids, math.max( fids, fide - seqLength + 1 ) )
		local fide = math.min( fide, fids + seqLength - 1 )
		local rw = torch.uniform(  )
		local rh = torch.uniform(  )
		local rf = torch.uniform(  )
		for f = 1, seqLength do
			local fid = math.min( fids + f - 1, fide )
			local fpath = paths.concat( vpath, string.format( self.frameFormat, fid - ffid + 1 ) )
			table.insert( dataTable, self:sampleHookTrain( fpath, rw, rh, rf ) )
			table.insert( labelTable, self.fid2cids[ fid ] )
		end
	end
	local data, labels = tableToOutput( self, dataTable, labelTable )
	return data, labels
end
function dataset:getActions( aidStart, quantity, seqLength )
	assert( quantity > 0 )
	assert( ( quantity % seqLength ) == 0 )
	local dataTable = {  }
	local labelTable = {  }
	local numActionToSample = quantity / seqLength
	for a = 1, numActionToSample do
		local aid = aidStart + a - 1
		local vid = self.aid2vid[ aid ]
		local vpath = ffi.string( torch.data( self.vid2path[ vid ] ) )
		local fids = self.aid2loc[ aid ][ 1 ]
		local fide = self.aid2loc[ aid ][ 2 ]
		local ffid = self.vid2ffid[ vid ]
		local fids = torch.random( fids, math.max( fids, fide - seqLength + 1 ) )
		local fide = math.min( fide, fids + seqLength - 1 )
		for f = 1, seqLength do
			local fid = math.min( fids + f - 1, fide )
			local fpath = paths.concat( vpath, string.format( self.frameFormat, fid - ffid + 1 ) )
			table.insert( dataTable, self:sampleHookTest( fpath ) )
			table.insert( labelTable, self.fid2cids[ fid ] )
		end
	end
	local data, labels = tableToOutput( self, dataTable, labelTable )
	return data, labels
end
function dataset:getVideo( vid )
	local numFrame = self.vid2numim[ vid ]
	local video = torch.Tensor( numFrame, 3, self.sampleSize, self.sampleSize )
	local vpath = ffi.string( torch.data( self.vid2path[ vid ] ) )
	for f = 1, numFrame do
		local fpath = paths.concat( vpath, string.format( self.frameFormat, f ) )
		local out = self:sampleHookTest( fpath )
		video[ f ]:copy( out )
	end
	local labels = self.vid2cids[ vid ][ self.vid2cids[ vid ]:gt( 0 ) ]
	return video, labels
end
evaluateBatch = function( fid2out, fid2cids, seqLength )
	local numFrame = fid2out:size( 1 )
	local numDim = fid2out:size( 2 )
	local map = 0
	for f = 1, numFrame do
		local cids = fid2cids[ f ][ fid2cids[ f ]:gt( 0 ) ]
		local _, rank2pcid = fid2out[ f ]:sort( true )
		local ap = 0
		local numTrue = 0
		local rank = 0
		for r = 1, numDim do
			numTrue = numTrue + cids:eq( rank2pcid[ r ] ):sum(  )
			ap = ap + numTrue / r
			if cids:size( 1 ) == numTrue then rank = r break end
		end
		map = map + ap / rank
	end
	map = map * 100 / numFrame
	return map
end
evaluateVideo = function( outputs, labels, features, seqLength )
	local numFrame = outputs:size( 1 )
	local numCls = outputs:size( 2 )
	local dimFeat = features[ 1 ]:numel(  )
	local numSeq = numFrame / seqLength
	local labels = labels[ labels:gt( 0 ) ]
	local topn = math.min( 10, numCls )
	local seqPool = 'sum'
	local vidPool = 'sum'
	assert( numSeq % 1 == 0 )
	-- Sequence-level pooling.
	local sid2out = torch.Tensor( numSeq, numCls )
	local sid2feat = torch.Tensor( numSeq, dimFeat )
	for s = 1, numSeq do
		local fstart = ( s - 1 ) * seqLength + 1
		local fend = fstart + seqLength - 1
		if seqPool == 'sum' then
			sid2out[ s ] = outputs[ { { fstart, fend } } ]:sum( 1 )
			sid2feat[ s ] = features[ { { fstart, fend } } ]:view( -1, dimFeat ):sum( 1 )
		elseif seqPool == 'max' then
			sid2out[ s ] = outputs[ { { fstart, fend } } ]:max( 1 )
			sid2feat[ s ] = features[ { { fstart, fend } } ]:view( -1, dimFeat ):max( 1 )
		elseif seqPool == 'last' then
			sid2out[ s ] = outputs[ fend ]
			sid2feat[ s ] = features[ fend ]:view( -1, dimFeat )
		end
	end
	-- Video-level pooling.
	local cid2score = nil
	local feature = nil
	if vidPool == 'sum' then
		cid2score = sid2out:sum( 1 )
		feature = sid2feat:sum( 1 )
	elseif vidPool == 'max' then
		cid2score = sid2out:max( 1 )
		feature = sid2feat:max( 1 )
	end
	-- Make and evaluate predictions.
	local _, rank2pcid = cid2score:sort( true )
	local topPreds = rank2pcid[ 1 ][ { { 1, topn } } ]
	local ap = 0
	local numTrue = 0
	local rank = 0
	for r = 1, numCls do
		numTrue = numTrue + labels:eq( rank2pcid[ 1 ][ r ] ):sum(  )
		ap = ap + numTrue / r
		if labels:size( 1 ) == numTrue then rank = r break end
	end
	ap = ap / rank
	return ap, topPreds, cid2score, feature
end
return dataset
