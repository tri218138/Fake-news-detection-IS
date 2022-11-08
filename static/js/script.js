var opened_example_list = false;
var mode = 'text';

function toggle_mode() {
    let element = document.getElementById('textInput');
    if (mode == 'text') {
        element.setAttribute('rows', 5);
        element.setAttribute('placeholder', 'Nhập đường dẫn ...');
        mode = 'url';
    }
    else {
        element.setAttribute('rows', 20);
        element.setAttribute('placeholder', 'Nội dung bài báo ...');
        mode = 'text';
    }
}

function clear_area_content() {
    document.getElementById('textInput').value = '';
    document.getElementById('textInput').style.color = 'black';
}

function show_example_list() {
    if (opened_example_list) {
        document.getElementById('examples').style.visibility = 'hidden';
        opened_example_list = false;
    }
    else {
        document.getElementById('examples').style.visibility = 'visible';
        opened_example_list = true;
    }
}

example_input = [
    [`Dueling Drills Further Increase NATO Russia Tensions November 03, 2016 Dueling Drills Further Increase NATO Russia Tensions 
Irked by NATO's disaster-relief drills in Montenegro, Russia held special forces exercises with its Balkan neighbour Serbia on Wednesday. In Montenegro, which split from Serbia a decade ago, NATO is on Friday due to wrap up a five-day disaster-relief drill of 680 unarmed personnel from seven NATO members and 10 partner countries including Serbia. ""It is in the interest of Montenegro, in the interest of NATO, and also in the interest of the region, to be part of NATO and promote Euro-Atlantic values in the Balkans, and these values benefit each and every one of us,"" Montenegro's deputy Prime Minister Dusko Markovic , said about the exercise. Across the border, in Serbia, Russia flew in its paratroopers along with allied Belarus for an exercise called ""Slavic Brotherhood"", which according the Serbian defence ministry focuses on ""fighting terrorism"". 
The drills are set to last until November 9 and will include 212 Russian, 450 Serbian and 56 Belarusian troops. Three Russian transport planes have already landed in Belgrade on Wednesday bringing in soldiers and equipment. 
Drills in the two former Yugoslav republics reflect mounting tensions between Russia and the West over a variety of geopolitical issues including conflicts in Syria and Ukraine. 
READ MORE: RUSSIA IS MOVING NUKES WEST IN RESPONSE TO CONTINUING NATO AGGRESSION 
In Serbia, NATO is hugely unpopular after its 1999 bombing ousted Belgrade's armed forces out of its former southern province of Kosovo. 
But as part of its integrations with the West, Serbia last year signed the Individual Partnership Action Plan (IPAP), the highest level of cooperation between NATO and a country not aspiring to join. 
An EU candidate, Serbia insists that it is militarily neutral, but maintains close ties with Russia, its Orthodox Christian and Slavic superpower ally, the biggest gas and oil source and an obstacle to Kosovo's membership in the United Nations. 
Last year Montenegro accepted an invitation to join NATO, a move that soured relations withRussia. Ahead of October's parliamentary vote Montenegro's outgoing Prime Minister Milo Djukanovic accused Moscow of meddling in country's elections and of financing the opposition. A fifth of the votes in the poll went to an anti-NATO, pro-Serbia bloc. Article by Doc Burkhart , Vice-President, General Manager and co-host of TRUNEWS with Rick Wiles Got a news tip? Email us at Help support the ministry of TRUNEWS with your one-time or monthly gift of financial support. DONATE NOW ! DOWNLOAD THE TRUNEWS MOBILE APP! CLICK HERE! Donate Today! Support TRUNEWS to help build a global news network that provides a credible source for world news 
We believe Christians need and deserve their own global news network to keep the worldwide Church informed, and to offer Christians a positive alternative to the anti-Christian bigotry of the mainstream news media Top Stories`
        , 1],
    [`A week before Michael T. Flynn resigned as national security adviser, a sealed proposal was to his office, outlining a way for President Trump to lift sanctions against Russia. Mr. Flynn is gone, having been caught lying about his own discussion of sanctions with the Russian ambassador. But the proposal, a peace plan for Ukraine and Russia, remains, along with those pushing it: Michael D. Cohen, the presidentâ€™s personal lawyer, who delivered the document Felix H. Sater, a business associate who helped Mr. Trump scout deals in Russia and a Ukrainian lawmaker trying to rise in a political opposition movement shaped in part by Mr. Trumpâ€™s former campaign manager Paul Manafort. At a time when Mr. Trumpâ€™s ties to Russia, and the people connected to him, are under heightened scrutiny  â€”   with investigations by American intelligence agencies, the F. B. I. and Congress  â€”   some of his associates remain willing and eager to wade into   efforts behind the scenes. Mr. Trump has confounded Democrats and Republicans alike with his repeated praise for the Russian president, Vladimir V. Putin, and his desire to forge an   alliance. While there is nothing illegal about such unofficial efforts, a proposal that seems to tip toward Russian interests may set off alarms. The amateur diplomats say their goal is simply to help settle a grueling,   conflict that has cost 10, 000 lives. â€œWho doesnâ€™t want to help bring about peace?â€ Mr. Cohen asked. But the proposal contains more than just a peace plan. Andrii V. Artemenko, the Ukrainian lawmaker, who sees himself as a   leader of a future Ukraine, claims to have evidence  â€”   â€œnames of companies, wire transfersâ€  â€”   showing corruption by the Ukrainian president, Petro O. Poroshenko, that could help oust him. And Mr. Artemenko said he had received encouragement for his plans from top aides to Mr. Putin. â€œA lot of people will call me a Russian agent, a U. S. agent, a C. I. A. agent,â€ Mr. Artemenko said. â€œBut how can you find a good solution between our countries if we do not talk?â€ Mr. Cohen and Mr. Sater said they had not spoken to Mr. Trump about the proposal, and have no experience in foreign policy. Mr. Cohen is one of several Trump associates under scrutiny in an F. B. I. counterintelligence examination of links with Russia, according to law enforcement officials he has denied any illicit connections. The two others involved in the effort have somewhat questionable pasts: Mr. Sater, 50, a   pleaded guilty to a role in a stock manipulation scheme decades ago that involved the Mafia. Mr. Artemenko spent two and a half years in jail in Kiev in the early 2000s on embezzlement charges, later dropped, which he said had been politically motivated. While it is unclear if the White House will take the proposal seriously, the diplomatic freelancing has infuriated Ukrainian officials. Ukraineâ€™s ambassador to the United States, Valeriy Chaly, said Mr. Artemenko â€œis not entitled to present any alternative peace plans on behalf of Ukraine to any foreign government, including the U. S. administration. â€ At a security conference in Munich on Friday, Mr. Poroshenko warned the West against â€œappeasementâ€ of Russia, and some American experts say offering Russia any alternative to a    international agreement on Ukraine would be a mistake. The Trump administration has sent mixed signals about the conflict in Ukraine. But given Mr. Trumpâ€™s praise for Mr. Putin, John Herbst, a former American ambassador to Ukraine, said he feared the new president might be too eager to mend relations with Russia at Ukraineâ€™s expense  â€”   potentially with a plan like Mr. Artemenkoâ€™s. It was late January when the three men associated with the proposed plan converged on the Loews Regency, a luxury hotel on Park Avenue in Manhattan where business deals are made in a lobby furnished with leather couches, over martinis at the restaurant bar and in private conference rooms on upper floors. Mr. Cohen, 50, lives two blocks up the street, in Trump Park Avenue. A lawyer who joined the Trump Organization in 2007 as special counsel, he has worked on many deals, including a   tower in the republic of Georgia and a   mixed martial arts venture starring a Russian fighter. He is considered a loyal lieutenant whom Mr. Trump trusts to fix difficult problems. The F. B. I. is reviewing an unverified dossier, compiled by a former British intelligence agent and funded by Mr. Trumpâ€™s political opponents, that claims Mr. Cohen met with a Russian representative in Prague during the presidential campaign to discuss Russiaâ€™s hacking of Democratic targets. But the Russian official named in the report told The New York Times that he had never met Mr. Cohen. Mr. Cohen insists that he has never visited Prague and that the dossierâ€™s assertions are fabrications. (Mr. Manafort is also under investigation by the F. B. I. for his connections to Russia and Ukraine.) Mr. Cohen has a personal connection to Ukraine: He is married to a Ukrainian woman and once worked with relatives there to establish an ethanol business. Mr. Artemenko, tall and burly, arrived at the Manhattan hotel between visits to Washington. (His wife, he said, met the first lady, Melania Trump, years ago during their modeling careers, but he did not try to meet Mr. Trump.) He had attended the inauguration and visited Congress, posting on Facebook his admiration for Mr. Trump and talking up his peace plan in meetings with American lawmakers. He entered Parliament in 2014, the year that the former Ukrainian president Viktor Yanukovych fled to Moscow amid protests over his economic alignment with Russia and corruption. Mr. Manafort, who had been instrumental in getting Mr. Yanukovych elected, helped shape a political bloc that sprang up to oppose the new president, Mr. Poroshenko, a wealthy businessman who has taken a far tougher stance toward Russia and accused Mr. Putin of wanting to absorb Ukraine into a new Russian Empire. Mr. Artemenko, 48, emerged from the opposition that Mr. Manafort nurtured. (The two men have never met, Mr. Artemenko said.) Before entering politics, Mr. Artemenko had business ventures in the Middle East and real estate deals in the Miami area, and had worked as an agent representing top Ukrainian athletes. Some colleagues in Parliament describe him as corrupt, untrustworthy or simply insignificant, but he appears to have amassed considerable wealth. He has fashioned himself in the image of Mr. Trump, presenting himself as Ukraineâ€™s answer to a rising class of nationalist leaders in the West. He even traveled to Cleveland last summer for the Republican National Convention, seizing on the chance to meet with members of Mr. Trumpâ€™s campaign. â€œItâ€™s time for new leaders, new approaches to the governance of the country, new principles and new negotiators in international politics,â€ he wrote on Facebook on Jan. 27. â€œOur time has come!â€ Mr. Artemenko said he saw in Mr. Trump an opportunity to advocate a plan for peace in Ukraine  â€”   and help advance his own political career. Essentially, his plan would require the withdrawal of all Russian forces from eastern Ukraine. Ukrainian voters would decide in a referendum whether Crimea, the Ukrainian territory seized by Russia in 2014, would be leased to Russia for a term of 50 or 100 years. The Ukrainian ambassador, Mr. Chaly, rejected a lease of that kind. â€œIt is a gross violation of the Constitution,â€ he said in written answers to questions from The Times. â€œSuch ideas can be pitched or pushed through only by those openly or covertly representing Russian interests. â€ The reaction suggested why Mr. Artemenkoâ€™s project also includes the dissemination of â€œkompromat,â€ or compromising material, purportedly showing that Mr. Poroshenko and his closest associates are corrupt. Only a new government, presumably one less hostile to Russia, might take up his plan. Mr. Sater, a longtime business associate of Mr. Trumpâ€™s with connections in Russia, was willing to help Mr. Artemenkoâ€™s proposal reach the White House. Mr. Trump has sought to distance himself from Mr. Sater in recent years. If Mr. Sater â€œwere sitting in the room right now,â€ Mr. Trump said in a 2013 deposition, â€œI really wouldnâ€™t know what he looked like. â€ But Mr. Sater worked on real estate development deals with the Trump Organization on and off for at least a decade, even after his role in the stock manipulation scheme came to light. Mr. Sater, who was born in the Soviet Union and grew up in New York, served as an executive at a firm called Bayrock Group, two floors below the Trump Organization in Trump Tower, and was later a senior adviser to Mr. Trump. He said he had been working on a plan for a Trump Tower in Moscow with a Russian real estate developer as recently as the fall of 2015, one that he said had come to a halt because of Mr. Trumpâ€™s presidential campaign. (Mr. Cohen said the Trump Organization had received a letter of intent for a project in Moscow from a Russian real estate developer at that time but determined that the project was not feasible.) Mr. Artemenko said a mutual friend had put him in touch with Mr. Sater. Helping to advance the proposal, Mr. Sater said, made sense. â€œI want to stop a war, number one,â€ he said. â€œNumber two, I absolutely believe that the U. S. and Russia need to be allies, not enemies. If I could achieve both in one stroke, it would be a home run. â€ After speaking with Mr. Sater and Mr. Artemenko in person, Mr. Cohen said he would deliver the plan to the White House. Mr. Cohen said he did not know who in the Russian government had offered encouragement on it, as Mr. Artemenko claims, but he understood there was a promise of proof of corruption by the Ukrainian president. â€œFraud is never good, right?â€ Mr. Cohen said. He said Mr. Sater had given him the written proposal in a sealed envelope. When Mr. Cohen met with Mr. Trump in the Oval Office in early February, he said, he left the proposal in Mr. Flynnâ€™s office. Mr. Cohen said he was waiting for a response when Mr. Flynn was forced from his post. Now Mr. Cohen, Mr. Sater and Mr. Artemenko are hoping a new national security adviser will take up their cause. On Friday the president wrote on Twitter that he had four new candidates for the job.`
, 0]
]

function get_example(ith) {
    let index = ith - 1;
    document.getElementById('textInput').value = example_input[index][0];
    show_example_list();
}

function predict_function() {
    let content = document.getElementById('textInput');
    //////////////// model predict ////////////////////
    res = '';
    if (content.value[0] == 'H') {
        if (example_input[0][1] == 1) {
            res = 'Tin thật';
            content.style.color = 'green';
        }
    }
    else if (content.value[0] == 'E') {
        if (example_input[1][1] == 0) {
            res = 'Tin giả';
            content.style.color = 'red';
        }
    }
    else {
        res = 'Nhập nội dung bài viết';
    }
    ////////////// visulize ///////////////////
    let element = document.getElementById('result');
    element.style.visibility = 'visible';
    element.innerHTML = '<strong>' + res + '</strong>';
}