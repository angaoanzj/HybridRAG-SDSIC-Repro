"""Candidate positions; head+tail is a union of original token positions."""
def positions(length,window):
    if window=='head180':return list(range(min(length,180)))
    if window=='tail180':return list(range(max(0,length-180),length))
    if window=='head180_tail180':return sorted(set(range(min(length,180)))|set(range(max(0,length-180),length)))
    if window=='full':return list(range(length))
    raise ValueError(window)

def check_rules():
    for length in [0,1,100,180,181,250,360,361,500]:
        head=positions(length,'head180');tail=positions(length,'tail180');both=positions(length,'head180_tail180');full=positions(length,'full')
        assert head==list(range(min(length,180))) and len(head)==min(length,180)
        assert tail==list(range(max(0,length-180),length))
        assert both==sorted(set(head)|set(tail)) and len(both)==min(length,360)
        assert full==list(range(length))
        if length<=180:assert head==tail==both==full
        if length<=360:assert both==full
    assert positions(500,'head180_tail180')==list(range(180))+list(range(320,500))
    return {'edge_lengths':9,'overlap_deduplicated':True,'original_order_preserved':True}

if __name__=='__main__':print(check_rules())
