async function loadAndDisplayPosts(json_filename, id, section_num) {
    try {
        // JSON 파일 로드
        const response = await fetch(json_filename);
        const data = await response.json();

        // 날짜 기준으로 오름차순 정렬
        data.htmls.sort((a, b) => a.time - b.date);

        // posts-container div에 게시글 링크 추가
        const container = document.getElementById(id);
        data.htmls.forEach((post, index) => {
            const anchor = document.createElement('a');
            const textNode = document.createTextNode(`${section_num}.${index+1}. `);
            const breakline = document.createElement('br');
            anchor.href = post.href;
            anchor.textContent = `${post.title}`;
            anchor.style.display = 'inline'; 
            container.appendChild(textNode);
            container.appendChild(anchor);
            container.appendChild(breakline);

        });
        console.log('Success')
    } catch (error) {
        console.error('게시글을 로드하는 데 실패했습니다:', error);
    }
}

async function loadAndDisplayPostsWithCategory(json_filename, id, section_num, category) {
    try {
        // JSON 파일 로드
        const response = await fetch(json_filename);
        const data = await response.json();

        // 날짜 기준으로 오름차순 정렬
        data.htmls.sort((a, b) => a.time - b.date);

        // posts-container div에 게시글 링크 추가
        const container = document.getElementById(id);
        index = 0;
        data.htmls.forEach(post => {
            if (post.category == category){
                const anchor = document.createElement('a');
                const textNode = document.createTextNode(`${section_num}.${index+1}. `);
                const breakline = document.createElement('br');
                anchor.href = post.href;
                anchor.textContent = `${post.title}`;
                anchor.style.display = 'inline'; 
                container.appendChild(textNode);
                container.appendChild(anchor);
                container.appendChild(breakline);
                index += 1;
            }
        });
        console.log('Success')
    } catch (error) {
        console.error('게시글을 로드하는 데 실패했습니다:', error);
    }
}